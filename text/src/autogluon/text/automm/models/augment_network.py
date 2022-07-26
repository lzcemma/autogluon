import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .utils import init_weights
from .mlp import Unit

class VAETransformer(nn.Module):
    def __init__(self, config: DictConfig, in_feautres: int, n_modality: int) -> None:
        super().__init__()
        self.config = config
        self.emb_d = in_feautres
        self.n_modality = n_modality
        print(f" VAE Transformer # features {n_modality}, dim {self.emb_d}")

        # learn augmentation prompt
        if self.config.n_emb != 0:
            self.emb = nn.Embedding(config.n_emb, in_feautres)
            self.emb_dropout = nn.Dropout(config.emb_dropout)
            self.emb_idx = nn.Parameter(torch.arange(0, config.n_emb, dtype=torch.int), requires_grad=False)

        # encoder
        encoder_layers = TransformerEncoderLayer(self.emb_d, config.n_head, config.tran_hidden, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_layer)

        # encoder linear z
        self.encoder_fc_z_mu = nn.Linear(self.emb_d, self.config.z_dim)
        self.encoder_fc_z_logvar = nn.Linear(self.emb_d, self.config.z_dim)

        # decoder linezr z
        self.decoder_fc = nn.Linear(self.config.z_dim, self.emb_d)

        # decoder
        decoder_layers = TransformerEncoderLayer(self.emb_d, config.n_head, config.tran_hidden, norm_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layers, config.n_layer)

        self.last_layer = nn.Linear(self.emb_d, self.emb_d)

        if self.config.gating == "sigmoid":
            self.gating = nn.Sigmoid()
        else:
            self.gating = nn.Identity()
        print(self.gating)
        self.init_parameters()

    def init_parameters(self):
        self.last_layer.weight.data.zero_()
        self.last_layer.bias.data.zero_()
        if self.config.n_emb != 0:
            self.emb.weight.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        input = X.reshape(-1, self.n_modality, self.emb_d)  # [B, # modality, emb dim] torch.Size([8, 3, 1024])

        if self.config.n_emb != 0:
            idx = self.emb_idx.repeat(len(X), 1)  # [B, # emb]
            emb = self.emb_dropout(self.emb(idx))  # [B, # emb, emb dim] torch.Size([8, 16, 1024])
            input = torch.cat([input, emb], dim=1)  # [B, # modality + # emb, emb dim]  torch.Size([8, 19, 1024])

        hidden = self.transformer_encoder(input)

        z_mu, z_logvar = self.encoder_fc_z_mu(hidden), self.encoder_fc_z_logvar(hidden)

        z = self.reparameterize(z_mu, z_logvar)

        hidden = self.decoder_fc(z)

        noise = self.gating(self.last_layer(self.transformer_decoder(hidden)[:, : self.n_modality, :]))
        recon_x = X.reshape(-1, self.n_modality, self.emb_d) + noise

        if self.config.n_emb != 0 and self.config.queue:
            new_emb = torch.cat([self.emb.weight.data, X.reshape(-1, self.emb_d)])
            self.emb.weight.data = new_emb[-self.config.n_emb :]

        return recon_x.reshape(len(X), -1), z_mu, z_logvar


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim=16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # Encoder P(Z|X)
        encoder_layers = []
        dims = [input_dim] + hidden_dim
        for i in range(len(dims) - 1):
            encoder_layers.append(
                Unit(
                    normalization="layer_norm",
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation="relu",
                    dropout_prob=0.5,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_fc_z_mu = nn.Linear(self.hidden_dim[-1], self.z_dim)
        self.encoder_fc_z_logvar = nn.Linear(self.hidden_dim[-1], self.z_dim)

        # Decoder P(X|Z)
        decoder_layers = []
        dims = [input_dim] + hidden_dim + [z_dim]

        for i in range(len(dims) - 1, 0, -1):
            decoder_layers.append(
                Unit(
                    normalization="layer_norm",
                    in_features=dims[i],
                    out_features=dims[i - 1],
                    activation="relu",
                    dropout_prob=0.5,
                )
            )
        self.decoder = nn.Sequential(*decoder_layers)
    
        self.init_parameters()

    def init_parameters(self):
        self.decoder[-1].fc.weight.data.zero_()
        self.decoder[-1].fc.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        z_mu, z_logvar = self.encoder_fc_z_mu(hidden), self.encoder_fc_z_logvar(hidden)
        z = self.reparameterize(z_mu, z_logvar)

        noise_x = self.decoder(z)
        recon_x = x + noise_x
        return recon_x, z_mu, z_logvar

class AugmentNetwork(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        feature_dims: Optional[List[Tuple]],
        adapter_out_dim: Optional[int],
        n_modality: Optional[int],
    ) -> None:
        super().__init__()
        print("Initializaing Augmentation Network")

        self.config = config
        self.feature_dims = feature_dims  # [('hf_text', 768), ('timm_image', 1024), ('clip', 512)]
        self.adapter_out_dim = adapter_out_dim
        print("feature_dims", self.feature_dims)
        print(f"cross_modality: {config.cross_modality}")
        print("after adapter dim", adapter_out_dim )
        if config.cross_modality:
            if config.arch == "mlp_vae":
                d = adapter_out_dim * len(feature_dims)
                step = int((d - self.config.z_dim) / (self.config.n_layer + 1))
                hidden = [*range(d - step, self.config.z_dim + step, -step)]
                self.augnets = VAE(input_dim=d, hidden_dim=hidden, z_dim=self.config.z_dim)
            elif config.arch == "trans_vae":
                self.augnets = VAETransformer(config, adapter_out_dim, n_modality)
            else:
                raise NotImplementedError
        else:
            if config.arch == "mlp_vae":
                self.augnets = nn.ModuleList()
                for i in range(n_modality):
                    d = adapter_out_dim
                    step = int((d - self.config.z_dim) / (self.config.n_layer + 1))
                    hidden = [*range(d - step, self.config.z_dim + step, -step)]
                    self.augnets.append(VAE(input_dim=d, hidden_dim=hidden, z_dim=self.config.z_dim))
            else:
                raise NotImplementedError
        self.name_to_id = self.get_layer_ids()

    def forward(self, x):
        if self.config.cross_modality:
            return self.augnets(x)
        else:
            x_splitted = torch.split(x, self.adapter_out_dim, dim = 1)

            new = []
            m = []
            v = []
            for i in range(len(x_splitted)):
                x_, m_, v_ = self.augnets[i](x_splitted[i])
                new += [x_]
                m += [m_]
                v += [v_]

            new = torch.cat(new, dim = 1)
            m = torch.cat(m, dim = 1)
            v = torch.cat(v, dim = 1)
            return new, m, v

    def get_layer_ids(
        self,
    ):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id

    def l2_regularize(self, x, x_new):
        return F.mse_loss(x_new, x, reduction="mean")

    def kld(self, m, v):
        return -0.5 * torch.sum(1 + v - m.pow(2) - v.exp())
