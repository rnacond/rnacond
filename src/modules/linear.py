import torch
from torch import nn
from typing import Optional


class FeedFoward(nn.Module):
    """Two linear layers separated by a non-linearity, typically used after attention"""

    def __init__(self, embedding_dim: int, ff_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        """Constructor

        Args:
            embedding_dim (int): embedding dimension.
            ff_dim (int, optional): hidden layer dimension. Defaults to None in which case it's 4 x embedding_dim.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        ff_dim = 4 * embedding_dim if ff_dim is None else ff_dim
        self.net = nn.Sequential(nn.Linear(embedding_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embedding_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input of size (batch_size, seq_len, embedding_dim)
        Returns:
            torch.Tensor: out of size (batch_size, seq_len, embedding_dim)
        """
        return self.net(x)
    
    
class ProjectionHead(nn.Module):
    """Projection head, typically used to project embeddings into another dimension """
    
    def __init__(self, embedding_dim: int, projection_dim: int = 256, dropout: float = 0.1) -> None:
        """Constructor

        Args:
            embedding_dim (int): embedding (input) dim 
            projection_dim (int, optional): Projection (output) dim. Defaults to 256
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x