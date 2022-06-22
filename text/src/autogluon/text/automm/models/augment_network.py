import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


class AugTransformer(nn.Module):
    def __init__(self, config: DictConfig, in_feautres: int, n_modality: int) -> None:
        super().__init__()
        self.config = config
        self.emb_d = in_feautres
        print(f"# features {n_modality}, dim {self.emb_d}")

        self.n_modality = n_modality

        encoder_layers = TransformerEncoderLayer(self.emb_d, config.n_head, config.tran_hidden)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_layer)

        self.emb = nn.Embedding(config.n_emb, in_feautres)
        self.emb_idx = nn.Parameter(torch.arange(0, config.n_emb, dtype=torch.int), requires_grad=False)

        self.name_to_id = self.get_layer_ids()

    def init_emb(self):
        self.emb.weight.data.zero_()

    def forward(self, X):
        X = X.reshape(-1, self.n_modality, self.emb_d)  # [B, # modality, emb dim] torch.Size([8, 3, 1024])

        idx = self.emb_idx.repeat(len(X), 1)  # [B, # emb]
        emb = self.emb(idx)  # [B, # emb, emb dim] torch.Size([8, 16, 1024])

        input = torch.cat([X, emb], dim=1)  # [B, # modality + # emb, emb dim]  torch.Size([8, 19, 1024])
        output = self.transformer_encoder(input)  # [B, # modality + # emb, emb dim]  torch.Size([8, 19, 1024])

        aug_output = output[:, : self.n_modality, :]  # [B, # modality, emb_dim] torch.Size([8, 3, 1024])

        return aug_output.reshape(len(X), -1)

    def l2_regularize(self, x, x_new):
        return F.mse_loss(x_new, x, reduction="mean")

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
