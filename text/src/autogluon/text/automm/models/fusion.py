from audioop import mul
import logging
import torch
from torch import detach, nn
from typing import List, Optional
from .utils import init_weights
from ..constants import LABEL, LOGITS, FEATURES, WEIGHT, AUTOMM
from .mlp import MLP
from .ft_transformer import FT_Transformer, CLSToken
from omegaconf import OmegaConf, DictConfig
from .augment_vae import MultiModalAugmentation
from .augment_network import AugTransformer, AugmentNetwork

logger = logging.getLogger(AUTOMM)


class MultimodalFusionMLP(nn.Module):
    """
    Use MLP to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through MLP.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: List[int],
        num_classes: int,
        adapt_in_features: Optional[str] = None,
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
        loss_weight: Optional[float] = None,
        aug_config: Optional[DictConfig] = None,
    ):
        """
        Parameters
        ----------
        prefix
            The fusion model's prefix
        models
            The individual models whose output features will be fused.
        hidden_features
            A list of integers representing the hidden feature dimensions. For example,
            [512, 128, 64] indicates three hidden MLP layers with their corresponding output
            feature dimensions.
        num_classes
            The number of classes.
        adapt_in_features
            Choice of how to adapt the features of each model. We now support
            - min
                Adapt all features to the minimum dimension. For example, if three models have
                feature dimensions [512, 768, 64], it will linearly map all the features to
                dimension 64.
            - max
                Adapt all features to the maximum dimension. For example, if three models have
                feature dimensions are [512, 768, 64], it will linearly map all the features to
                dimension 768.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxilliary loss for each individual model.
        """
        super().__init__()
        logger.debug("initializing MultimodalFusionMLP")
        if loss_weight is not None:
            assert loss_weight > 0
        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)

        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])

            in_features = base_in_feat * len(raw_in_features)
            self.adapter_out_dim = base_in_feat
        else:
            self.adapter = nn.ModuleList([nn.Identity() for _ in range(len(raw_in_features))])
            in_features = sum(raw_in_features)
        assert len(self.adapter) == len(self.model)

        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        self.head = nn.Linear(in_features, num_classes)
        # adverasial trained augmentation network for features
        self.augmenter = None
        self.aug_config = aug_config
        if aug_config != None and aug_config.turn_on:
            self.augmenter = self.construct_augnet()
            self.pre_adapter = True
            if self.aug_config.cross_modality:
                self.pre_adapter = False
            print(f"pre adapter: {self.pre_adapter}")

        # init weights
        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)

        self.prefix = prefix

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def construct_augnet(self):
        model_feature_dict = [(per_model.prefix, per_model.out_features) for per_model in self.model]
        print(model_feature_dict)
        if self.aug_config.arch == "mlp_vae":
            return MultiModalAugmentation(self.aug_config, model_feature_dict)
        elif self.aug_config.arch == "trans" or self.aug_config.arch == "trans_vae":
            return AugmentNetwork(self.aug_config, model_feature_dict, self.adapter_out_dim, len(self.model))

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(self, batch: dict, is_training: Optional[bool]):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data. The fusion model doesn't need to
            directly access the mini-batch data since it aims to fuse the individual models'
            output features.

        Returns
        -------
        If "loss_weight" is None, it returns dictionary containing the fusion model's logits and
        features. Otherwise, it returns a list of dictionaries collecting all the models' output,
        including the fusion model's.
        """
        multimodal_output = {}
        for per_model in self.model:
            per_output = per_model(batch)
            multimodal_output[per_model.prefix] = per_output

        if self.aug_config.keep_original:

            aug_loss = None
            if self.augmenter is not None:
                if self.pre_adapter and is_training:
                    aug_loss = {}
                    for per_model in self.model:
                        k = per_model.prefix

                        detached_feature = multimodal_output[k][k][FEATURES].detach().clone()

                        if self.aug_config.arch == "mlp_vae":
                            new, m, v = self.augmenter(k, detached_feature)
                            regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                            KLD_loss = self.augmenter.kld(m, v) / new.size(0) / new.size(1)
                            aug_loss.update(
                                {
                                    k: {
                                        "regularizer": regularize_loss,
                                        "KLD_loss": KLD_loss,
                                        "reg_weight": self.aug_config.regularizer_loss_weight,
                                        "kl_weight": self.aug_config.kl_loss_weight,
                                    }
                                }
                            )
                        elif self.aug_config.arch == "trans":
                            new = self.augmenter(k, detached_feature)
                            regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                            aug_loss.update(
                                {
                                    k: {
                                        "regularizer": regularize_loss,
                                        "reg_weight": self.aug_config.regularizer_loss_weight,
                                    }
                                }
                            )

                        after_augment_logit = per_model.head(new)
                        new.register_hook(lambda grad: -grad * self.aug_config.adv_weight)

                        multimodal_output[k][k][FEATURES] = torch.cat([multimodal_output[k][k][FEATURES], new], dim=0)
                        # multimodal_output[k][k][LOGITS] = torch.cat(
                        #     [multimodal_output[k][k][LOGITS], after_augment_logit], dim=0
                        # )

                    # # to pass in fusion
                    # multimodal_output[k][k][FEATURES].register_hook(
                    #     lambda grad: -grad * (1 / self.aug_config.adv_weight)
                    # )
                    # if self.aug_config.arch == "n_vae":
                    #     multimodal_output[k][k][FEATURES], _, _ = self.augmenter(k, multimodal_output[k][k][FEATURES])
                    # elif self.aug_config == "trans":
                    #     multimodal_output[k][k][FEATURES], _, _ = self.augmenter(k, multimodal_output[k][k][FEATURES])
                    # multimodal_output[k][k][LOGITS] = per_model.head(multimodal_output[k][k][FEATURES])
                    # multimodal_output[k][k][FEATURES].register_hook(lambda grad: -grad * self.aug_config.adv_weight)

            multimodal_features = []
            output = {}
            for per_model, per_adapter in zip(self.model, self.adapter):
                multimodal_features.append(
                    per_adapter(multimodal_output[per_model.prefix][per_model.prefix][FEATURES])
                )
                if self.loss_weight is not None:
                    multimodal_output[per_model.prefix][per_model.prefix].update({WEIGHT: self.loss_weight})
                    output.update(multimodal_output[per_model.prefix])
            multimodal_features = torch.cat(multimodal_features, dim=1)

            # pass through augmentation network after adapter
            if self.augmenter is not None:
                if is_training and self.pre_adapter is False:

                    # train augment network
                    aug_loss = {}
                    detached_feature = multimodal_features.detach().clone()
                    if self.aug_config.arch == "trans_vae":
                        new, m, v = self.augmenter(None, detached_feature)
                        regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                        KLD_loss = self.augmenter.kld(m, v) / new.size(0) / new.size(1)
                        aug_loss.update(
                            {
                                "transformer_augnet": {
                                    "regularizer": regularize_loss,
                                    "KLD_loss": KLD_loss,
                                    "reg_weight": self.aug_config.regularizer_loss_weight,
                                    "kl_weight": self.aug_config.kl_loss_weight,
                                }
                            }
                        )

                    elif self.aug_config.arch == "trans":
                        new = self.augmenter(None, detached_feature)
                        regularize_loss = self.augmenter.l2_regularize(detached_feature, new)

                        aug_loss.update(
                            {
                                "transformer_augnet": {
                                    "regularizer": regularize_loss,
                                    "reg_weight": self.aug_config.regularizer_loss_weight,
                                }
                            }
                        )

                    new.register_hook(lambda grad: -grad * (self.aug_config.adv_weight))
                    multimodal_features = torch.cat([multimodal_features, new], dim=0)

            features = self.fusion_mlp(multimodal_features)
            logits = self.head(features)
            fusion_output = {
                self.prefix: {
                    LOGITS: logits,
                    FEATURES: features,
                }
            }
            if self.loss_weight is not None:
                fusion_output[self.prefix].update({WEIGHT: 1})
                output.update(fusion_output)
            else:
                output = fusion_output

            if aug_loss is not None:
                output.update({"augmenter": aug_loss})

        else:
            aug_loss = None
            if self.augmenter is not None:
                if self.pre_adapter and is_training:

                    aug_loss = {}
                    for per_model in self.model:
                        k = per_model.prefix
                        # for kl, mse loss
                        detached_feature = multimodal_output[k][k][FEATURES].detach().clone()

                        if self.aug_config.arch == "mlp_vae":
                            new, m, v = self.augmenter(k, detached_feature)
                            regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                            KLD_loss = self.augmenter.kld(m, v) / new.size(0) / new.size(1)
                            aug_loss.update(
                                {
                                    k: {
                                        "regularizer": regularize_loss,
                                        "KLD_loss": KLD_loss,
                                        "reg_weight": self.aug_config.regularizer_loss_weight,
                                        "kl_weight": self.aug_config.kl_loss_weight,
                                    }
                                }
                            )
                        elif self.aug_config.arch == "trans":
                            new = self.augmenter(k, detached_feature)
                            regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                            aug_loss.update(
                                {
                                    k: {
                                        "regularizer": regularize_loss,
                                        "reg_weight": self.aug_config.regularizer_loss_weight,
                                    }
                                }
                            )

                        # augment to pass in fusion
                        if self.aug_config.original_ratio > 0.0:
                            orignal_multimodal_features, for_augment_multimodal_features = torch.split(
                                multimodal_output[k][k][FEATURES],
                                int(self.aug_config.original_ratio * len(multimodal_output[k][k][FEATURES])),
                            )
                            orignal_logit, after_augment_logit = torch.split(
                                multimodal_output[k][k][LOGITS],
                                int(self.aug_config.original_ratio * len(multimodal_output[k][k][FEATURES])),
                            )
                        else:
                            orignal_multimodal_features, for_augment_multimodal_features = (
                                None,
                                multimodal_output[k][k][FEATURES],
                            )
                            orignal_logit, after_augment_logit = (
                                None,
                                multimodal_output[k][k][LOGITS],
                            )

                        for_augment_multimodal_features.register_hook(
                            lambda grad: -grad * (1 / self.aug_config.adv_weight)
                        )
                        if self.aug_config.arch == "n_vae":
                            for_augment_multimodal_features, _, _ = self.augmenter(k, for_augment_multimodal_features)
                        elif self.aug_config.arch == "trans":
                            for_augment_multimodal_features = self.augmenter(k, for_augment_multimodal_features)
                        after_augment_logit = per_model.head(for_augment_multimodal_features)
                        for_augment_multimodal_features.register_hook(lambda grad: -grad * self.aug_config.adv_weight)

                        if orignal_multimodal_features is not None:
                            multimodal_output[k][k][FEATURES] = torch.cat(
                                [orignal_multimodal_features, for_augment_multimodal_features], dim=0
                            )
                            multimodal_output[k][k][LOGITS] = torch.cat([orignal_logit, after_augment_logit], dim=0)
                        else:
                            multimodal_output[k][k][FEATURES] = for_augment_multimodal_features
                            multimodal_output[k][k][LOGITS] = after_augment_logit

                        # # to pass in fusion
                        # multimodal_output[k][k][FEATURES].register_hook(
                        #     lambda grad: -grad * (1 / self.aug_config.adv_weight)
                        # )
                        # if self.aug_config.arch == "n_vae":
                        #     multimodal_output[k][k][FEATURES], _, _ = self.augmenter(k, multimodal_output[k][k][FEATURES])
                        # elif self.aug_config == "trans":
                        #     multimodal_output[k][k][FEATURES], _, _ = self.augmenter(k, multimodal_output[k][k][FEATURES])
                        # multimodal_output[k][k][LOGITS] = per_model.head(multimodal_output[k][k][FEATURES])
                        # multimodal_output[k][k][FEATURES].register_hook(lambda grad: -grad * self.aug_config.adv_weight)

            multimodal_features = []
            output = {}
            for per_model, per_adapter in zip(self.model, self.adapter):
                multimodal_features.append(
                    per_adapter(multimodal_output[per_model.prefix][per_model.prefix][FEATURES])
                )
                if self.loss_weight is not None:
                    multimodal_output[per_model.prefix][per_model.prefix].update({WEIGHT: self.loss_weight})
                    output.update(multimodal_output[per_model.prefix])
            multimodal_features = torch.cat(multimodal_features, dim=1)

            # pass through augmentation network after adapter
            if self.augmenter is not None:
                if is_training and self.pre_adapter is False:

                    # train augment network
                    aug_loss = {}
                    detached_feature = multimodal_features.detach().clone()
                    if self.aug_config.arch == "trans_vae":
                        new, m, v = self.augmenter(None, detached_feature)
                        regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                        KLD_loss = self.augmenter.kld(m, v) / new.size(0) / new.size(1)
                        aug_loss.update(
                            {
                                "transformer_augnet": {
                                    "regularizer": regularize_loss,
                                    "KLD_loss": KLD_loss,
                                    "reg_weight": self.aug_config.regularizer_loss_weight,
                                    "kl_weight": self.aug_config.kl_loss_weight,
                                }
                            }
                        )

                    elif self.aug_config.arch == "trans":
                        x_new = self.augmenter(None, detached_feature)
                        regularize_loss = self.augmenter.l2_regularize(detached_feature, x_new)

                        aug_loss.update(
                            {
                                "transformer_augnet": {
                                    "regularizer": regularize_loss,
                                    "reg_weight": self.aug_config.regularizer_loss_weight,
                                }
                            }
                        )

                    # augment to pass in fusion
                    if self.aug_config.original_ratio > 0.0:
                        orignal_multimodal_features, for_augment_multimodal_features = torch.split(
                            multimodal_features, int(self.aug_config.original_ratio * len(multimodal_features))
                        )
                    else:
                        orignal_multimodal_features, for_augment_multimodal_features = None, multimodal_features

                    for_augment_multimodal_features.register_hook(
                        lambda grad: -grad * (1 / self.aug_config.adv_weight)
                    )
                    if self.aug_config.arch == "trans_vae":
                        for_augment_multimodal_features, _, _ = self.augmenter(None, for_augment_multimodal_features)
                    elif self.aug_config.arch == "trans":
                        for_augment_multimodal_features = self.augmenter(None, for_augment_multimodal_features)
                    for_augment_multimodal_features.register_hook(lambda grad: -grad * self.aug_config.adv_weight)

                    if orignal_multimodal_features is not None:
                        multimodal_features = torch.cat(
                            [orignal_multimodal_features, for_augment_multimodal_features], dim=0
                        )
                    else:
                        multimodal_features = for_augment_multimodal_features

            features = self.fusion_mlp(multimodal_features)
            logits = self.head(features)
            fusion_output = {
                self.prefix: {
                    LOGITS: logits,
                    FEATURES: features,
                }
            }
            if self.loss_weight is not None:
                fusion_output[self.prefix].update({WEIGHT: 1})
                output.update(fusion_output)
            else:
                output = fusion_output

            if aug_loss is not None:
                output.update({"augmenter": aug_loss})

        return output

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        names = [n for n, _ in self.named_parameters()]

        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]

        name_to_id = {}
        logger.debug(f"outer layers are treated as head: {outer_layer_names}")
        for n in outer_layer_names:
            name_to_id[n] = 0

        for i, per_model in enumerate(self.model):
            per_model_prefix = f"{model_prefix}.{i}"
            if not hasattr(per_model, "name_to_id"):
                raise ValueError(f"name_to_id attribute is missing in model: {per_model.__class__.__name__}")
            for n, layer_id in per_model.name_to_id.items():
                full_n = f"{per_model_prefix}.{n}"
                name_to_id[full_n] = layer_id

        # double check each parameter has been assigned an id
        for n in names:
            assert n in name_to_id

        return name_to_id


class MultimodalFusionTransformer(nn.Module):
    """
    Use Transformer to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through Transformer.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: int,
        num_classes: int,
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = "kaiming",
        attention_normalization: Optional[str] = "layer_norm",
        attention_dropout: Optional[str] = 0.2,
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = "reglu",
        ffn_normalization: Optional[str] = "layer_norm",
        ffn_d_hidden: Optional[str] = 192,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] = False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = "relu",
        head_normalization: Optional[str] = "layer_norm",
        adapt_in_features: Optional[str] = None,
        loss_weight: Optional[float] = None,
    ):
        super().__init__()
        logger.debug("initializing MultimodalFusionTransformer")
        if loss_weight is not None:
            assert loss_weight > 0

        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)

        raw_in_features = [per_model.out_features for per_model in models]

        if adapt_in_features == "min":
            base_in_feat = min(raw_in_features)
        elif adapt_in_features == "max":
            base_in_feat = max(raw_in_features)
        else:
            raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

        self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])

        in_features = base_in_feat

        assert len(self.adapter) == len(self.model)

        self.fusion_transformer = FT_Transformer(
            d_token=in_features,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization=attention_initialization,
            attention_normalization=attention_normalization,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=ffn_normalization,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            first_prenormalization=first_prenormalization,
            last_layer_query_idx=None,
            n_tokens=None,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=hidden_features,
            projection=False,
        )

        self.head = FT_Transformer.Head(
            d_in=in_features,
            d_out=num_classes,
            bias=True,
            activation=head_activation,
            normalization=head_normalization,
        )

        self.cls_token = CLSToken(
            d_token=in_features,
            initialization="uniform",
        )

        # init weights
        self.adapter.apply(init_weights)
        self.head.apply(init_weights)

        self.prefix = prefix

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []
        output = {}
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)

            if self.loss_weight is not None:
                per_output[per_model.prefix].update({WEIGHT: self.loss_weight})
                output.update(per_output)

        multimodal_features = torch.cat(multimodal_features, dim=1)
        multimodal_features = self.cls_token(multimodal_features)
        features = self.fusion_transformer(multimodal_features)

        logits = self.head(features)
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }
        if self.loss_weight is not None:
            fusion_output[self.prefix].update({WEIGHT: 1})
            output.update(fusion_output)
            return output
        else:
            return fusion_output

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        names = [n for n, _ in self.named_parameters()]

        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]
        name_to_id = {}
        logger.debug(f"outer layers are treated as head: {outer_layer_names}")
        for n in outer_layer_names:
            name_to_id[n] = 0

        for i, per_model in enumerate(self.model):
            per_model_prefix = f"{model_prefix}.{i}"
            if not hasattr(per_model, "name_to_id"):
                raise ValueError(f"name_to_id attribute is missing in model: {per_model.__class__.__name__}")
            for n, layer_id in per_model.name_to_id.items():
                full_n = f"{per_model_prefix}.{n}"
                name_to_id[full_n] = layer_id

        # double check each parameter has been assigned an id
        for n in names:
            assert n in name_to_id

        return name_to_id
