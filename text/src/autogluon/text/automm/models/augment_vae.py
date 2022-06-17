from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import Unit
from .utils import init_weights
from typing import List, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from ..constants import LABEL, LOGITS, FEATURES, WEIGHT, AUTOMM


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


class MultiModalAugmentation(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        feature_dims: List[Tuple],
    ) -> None:
        super().__init__()
        print("Initializaing Augmentation Network")

        self.config = config
        self.feature_dims = feature_dims
        self.augnets = nn.ModuleDict()
        for k, d in self.feature_dims:
            step = int((d - self.config.z_dim) / (self.config.n_layer + 1))
            hidden = [*range(d - step, self.config.z_dim + step, -step)]
            self.augnets.update([[k, VAE(input_dim=d, hidden_dim=hidden, z_dim=self.config.z_dim)]])
        self.augnets.apply(init_weights)

        for k in self.augnets.keys():
            self.augnets[k].decoder[-1].fc.weight.data.zero_()

        self.name_to_id = self.get_layer_ids()

    def forward(self, k, x):
        return self.augnets[k](x)

    def l2_regularize(self, x, x_new):
        return F.mse_loss(x_new, x, reduction="mean")

    def kld(self, m, v):
        return -0.5 * torch.sum(1 + v - m.pow(2) - v.exp())

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
