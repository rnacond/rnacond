import torch    
import torch.nn as nn
import math
from typing import Optional, Literal



class SinusoidalPosEnc(nn.Module):
    """Fixed sinusoidal positional encoding"""
    
    
    def __init__(self, dim: int, theta: int = 10000) -> None:
        """Constructor

        Args:
            dim (int): dimension of the encoding
            theta (int): theta
        """

        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """

        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb

class RandomOrLearnedSinusoidalPosEnc(nn.Module):
    """Random or learned sinusoidal positional encoding
    github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8"""

    def __init__(self, dim: int, is_random: bool = False) -> None:
        """Constructor
        
        Args:
            dim (int): dimension of the encoding
            is_random (bool): random or learned
        """
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward
        
        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """

        x = x.view(-1, 1)
        weights = self.weights.view(1, -1)
        freqs = x * weights * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class SinCosPosEncoding(nn.Module):
    """SinCos fixed positional Encoding"""

    def __init__(self, block_size: int, embedding_dim: int) -> None:
        """Constructor

        Args:
            block_size (int): block size
            embedding_dim (int): embedding dimension
        """

        super(SinCosPosEncoding, self).__init__()
        
        pe = torch.zeros(block_size, embedding_dim)
        position = torch.arange(0, block_size, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"Forward
        
        Args:
            x (torch.Tensor): input
        
        Returns:
            torch.Tensor: encoding
        """

        return x + self.pe[:, :x.size(-1)]
    
def get_position_embedding_layer(strat: Literal['embedding', 'encoding', 'none'], block_size: Optional[int] = None, embedding_dim: Optional[int] = None) -> Optional[nn.Module]:
    """Get positional encoding layer

    Args:
        strat (Literal['embedding', 'encoding', 'none']): strategy
        block_size (Optional[int]): block size
        embedding_dim (Optional[int]): embedding dimension

    Returns:
        Optional[nn.Module]: positional encoding layer
    """
    if strat == 'embedding':
        return nn.Embedding(block_size, embedding_dim)
    elif strat == 'encoding':
        return SinCosPosEncoding(block_size, embedding_dim)
    elif strat == 'none':
        return None
    else:
        raise ValueError("Invalid positional encoding strategy")