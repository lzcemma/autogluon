import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .utils import init_weights


class VAETransformer(nn.Module):
    def __init__(self, config: DictConfig, in_feautres: int, n_modality: int) -> None:
        super().__init__()
        self.config = config
        self.emb_d = in_feautres
        self.n_modality = n_modality
        print(f" VAE Transformer # features {n_modality}, dim {self.emb_d}")

        # memory slot
        if self.config.n_emb != 0:
            self.emb = nn.Embedding(config.n_emb, in_feautres)
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
        self.init_parameters()

    def init_parameters(self):
        self.last_layer.weight.data.zero_()
        self.last_layer.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        input = X.reshape(-1, self.n_modality, self.emb_d)  # [B, # modality, emb dim] torch.Size([8, 3, 1024])

        if self.config.n_emb != 0:
            idx = self.emb_idx.repeat(len(X), 1)  # [B, # emb]
            emb = self.emb(idx)  # [B, # emb, emb dim] torch.Size([8, 16, 1024])
            input = torch.cat([input, emb], dim=1)  # [B, # modality + # emb, emb dim]  torch.Size([8, 19, 1024])

        hidden = self.transformer_encoder(input)  # ([8, 2, 768])

        z_mu, z_logvar = self.encoder_fc_z_mu(hidden), self.encoder_fc_z_logvar(hidden)

        z = self.reparameterize(z_mu, z_logvar)

        hidden = self.decoder_fc(z)

        noise = self.last_layer(self.transformer_decoder(hidden)[:, : self.n_modality, :])
        recon_x = X.reshape(-1, self.n_modality, self.emb_d) + noise

        return recon_x.reshape(len(X), -1), z_mu, z_logvar


class AugTransformer(nn.Module):
    def __init__(self, config: DictConfig, in_feautres: int, n_modality: int) -> None:
        super().__init__()
        self.config = config
        self.emb_d = in_feautres
        print(f"Transformer: # features {n_modality}, dim {self.emb_d}")

        self.n_modality = n_modality

        encoder_layers = TransformerEncoderLayer(self.emb_d, config.n_head, config.tran_hidden, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_layer)
        if self.config.n_emb != 0:
            self.emb = nn.Embedding(config.n_emb, in_feautres)
            self.emb_idx = nn.Parameter(torch.arange(0, config.n_emb, dtype=torch.int), requires_grad=False)

        self.name_to_id = self.get_layer_ids()
        self.last_layer = nn.Linear(self.emb_d, self.emb_d)
        self.init_parameters()

    def init_parameters(self):
        self.last_layer.weight.data.zero_()
        self.last_layer.bias.data.zero_()

    def forward(self, X):
        input = X.reshape(-1, self.n_modality, self.emb_d)  # [B, # modality, emb dim] torch.Size([8, 3, 1024])

        if self.config.n_emb != 0:
            idx = self.emb_idx.repeat(len(X), 1)  # [B, # emb]
            emb = self.emb(idx)  # [B, # emb, emb dim] torch.Size([8, 16, 1024])
            input = torch.cat([input, emb], dim=1)  # [B, # modality + # emb, emb dim]  torch.Size([8, 19, 1024])
        output = self.transformer_encoder(input)  # [B, # modality + # emb, emb dim]  torch.Size([8, 19, 1024])

        aug_output = self.last_layer(output[:, : self.n_modality, :]) + X.reshape(
            -1, self.n_modality, self.emb_d
        )  # [B, # modality, emb_dim] torch.Size([8, 3, 1024])
        return aug_output.reshape(len(X), -1)

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
        print(f"cross_modality: {config.cross_modality}")
        if config.cross_modality:
            if config.arch == "trans":
                self.augnets = AugTransformer(config, adapter_out_dim, n_modality)
            elif config.arch == "trans_vae":
                self.augnets = VAETransformer(config, adapter_out_dim, n_modality)
        else:
            self.augnets = nn.ModuleDict()
            for k, d in self.feature_dims:
                self.augnets.update([[k, AugTransformer(config, d, 1)]])
        self.name_to_id = self.get_layer_ids()

    def forward(self, k, x):
        if self.config.cross_modality:
            return self.augnets(x)
        else:
            return self.augnets[k](x)

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
