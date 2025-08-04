from typing import Dict

from torch import nn, Tensor

from baseline.utils.network import Conv1dWithConstraint
from data.processor.wrapper import get_dataset_montage


class DynamicChannelConvRouter(nn.Module):
    def __init__(self, ds_conf: dict[str, str], target_channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds_conf = ds_conf
        self.target_channel = target_channel

        self.montage_dict: dict[str, int] = dict()
        self.collect_montage()

        self.conv_router = nn.ModuleDict()
        for mont_name, mont_len in self.montage_dict.items():
            self.add_conv(mont_name, mont_len)

    def collect_montage(self):
        for ds_name, conf_name in self.ds_conf.items():
            montages = get_dataset_montage(dataset_name=ds_name, config_name=conf_name)
            for mont_name, montage in montages.items():
                self.montage_dict.update({mont_name: len(montage)})

    def add_conv(self, mont_name: str, mont_len: int):
        self.conv_router[mont_name] = Conv1dWithConstraint(
                mont_len, self.target_channel, 1, max_norm=1
            )

    def forward(self, x: Tensor, mont_name: str) -> Tensor:
        if mont_name not in self.conv_router.keys():
            raise ValueError(f"Head '{mont_name}' not found. Available heads: {list(self.conv_router.keys())}")

        x = self.conv_router[mont_name](x)
        return x


class AdaptiveClassificationHead(nn.Module):
    """
    Adaptive classification head that can handle variable sequence lengths.
    Uses global average pooling to handle different sequence lengths.
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dims: list[int],
            n_class: int,
            dropout: float = 0.3,
            average_pooling: bool = True,
            activation = nn.ELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_class = n_class
        self.average_pooling = average_pooling

        # Global pooling to handle variable sequence lengths
        if self.average_pooling:
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        layers = []
        input_dim = embed_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(embed_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        # Add final output layer
        layers.append(nn.Linear(input_dim, n_class))

        # Combine all layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor, capture_features: bool = False) -> Tensor:
        """
        Args:
            x: Features of shape [B, seq_len, embed_dim]
            capture_features: Whether to return features for t-SNE
        Returns:
            logits: [B, n_class]
        """
        if self.average_pooling:
            # x: [B, seq_len, embed_dim] -> [B, embed_dim, seq_len]
            x = x.transpose(1, 2)
            # Global average pooling: [B, embed_dim, seq_len] -> [B, embed_dim, 1]
            x = self.global_pool(x)

            # Flatten: [B, embed_dim, 1] -> [B, embed_dim]
            x = x.squeeze(-1)

        # Fetch features for t-sne
        if capture_features:
            return x

        # Classification
        logits = self.mlp(x)
        return logits


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier that maintains separate classification heads
    for different datasets or montages.
    """

    def __init__(
            self,
            embed_dim: int,
            mlp_dims: list[int],
            head_configs: Dict[str, int],
            dropout: float = 0.5,
            average_pooling: bool = True,
            t_sne: bool = False,
    ):
        """
        Args:
            embed_dim: Feature dimension from encoder
            head_configs: Dict mapping head_name -> n_classes
            dropout: Dropout rate for classification heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_configs = head_configs
        self.mlp_dims = mlp_dims
        self.average_pooling = average_pooling
        self.dropout_rate = dropout

        # Create separate classification heads
        self.heads = nn.ModuleDict()
        for head_name, n_class in head_configs.items():
            self.add_head(head_name, n_class)

        self.cls_feature = None
        self.t_sne = t_sne

    def forward(self, x: Tensor, head_name: str) -> Tensor:
        """
        Forward pass using specified classification head.

        Args:
            x: Features of shape [B, seq_len, embed_dim]
            head_name: Name of the classification head to use

        Returns:
            logits: [B, n_class] for the specified head
        """
        if head_name not in self.heads:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {list(self.heads.keys())}")

        if self.t_sne:
            # Capture shared features for t-SNE visualization
            shared_features = self.heads[head_name](x, capture_features=True)
            self.cls_feature = shared_features.clone()
            
        # Continue with normal forward pass for logits
        logits = self.heads[head_name](x, capture_features=False)
        return logits

    def add_head(self, head_name: str, n_class: int):
        """Add a new classification head."""
        self.heads[head_name] = AdaptiveClassificationHead(
            embed_dim=self.embed_dim,
            hidden_dims=self.mlp_dims,
            n_class=n_class,
            dropout=self.dropout_rate,
            average_pooling=self.average_pooling
        )
        self.head_configs[head_name] = n_class

    def get_available_heads(self) -> list[str]:
        """Get a list of available classification heads."""
        return list(self.heads.keys())
