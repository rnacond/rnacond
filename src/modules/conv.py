import torch
import torch.nn as nn
from einops import rearrange, einsum
from src.utils.utils import exists
from typing import Tuple
from src.modules.layer_utils import RMSNorm




def Upsample(channels_in: int, channels_out = None) -> nn.Module:
    """Upsammpling followed by a convolution

    Args:
        channels_in (int): number of channels (filters) in
        channels_out (int): number of channels (filters) out
    Returns:
        nn.Module
    """

    channels_out = channels_in if channels_out is None else channels_out
    return nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'nearest'), nn.Conv1d(channels_in, channels_out, 3, padding = 1))

def Downsample(channels_in: int, channels_out = None) -> nn.Module:
    """Downsammpling via a convolution

    Args:
        channels_in (int): number of channels (filters) in
        channels_out (int): number of channels (filters) out
    Returns:
        nn.Module
    """
    channels_out = channels_in if channels_out is None else channels_out
    return nn.Conv1d(channels_in, channels_out, 4, 2, 1)



class ConvBlock(nn.Module):
    """Basic block containing a convolution a normalization layer and an activation"""

    def __init__(self, channels_in: int, channels_out: int, groups: int = 8) -> None:
        """Constructor

        Args:
            channels_in (int): number of channels (filters) in
            channels_out (int): number of channels (filters) out
            groups (int): number of conv groups
        """

        super().__init__()
        self.proj = nn.Conv1d(channels_in, channels_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, channels_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Tuple[float, float] = None) -> torch.Tensor:    
        """Forward function
        
        Args:
            x (torch.Tensor): input
            scale_shift (Tuple[float, float]): scale and shift
            
        Returns:
            torch.Tensor: output"""
        
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class TimeResidualConvBlock(nn.Module):
    """Residual block with time embedding containing two ConvBlock"""

    def __init__(self, channels_in: int, channels_out: int, time_emb_dim: int = 0, groups: int = 8) -> None:
        """Constructor

        Args:
            channels_in (int): number of channels (filters) in
            channels_out (int): number of channels (filters) out
            time_emb_dim (int): dimension of time embedding (no time embedding if 0)
            groups (int): number of groups
        """

        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels_out * 2)) if time_emb_dim is not None else None
        self.block1 = ConvBlock(channels_in, channels_out, groups = groups)
        self.block2 = ConvBlock(channels_out, channels_out, groups = groups)
        self.res_conv = nn.Conv1d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input
            time_emb (torch.Tensor): time embedding 

        Returns:
            torch.Tensor: output
        """

        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class CondTimeResidualConvBlock(nn.Module):
    """Residual block with time embedding qnd conditioning containing two ConvBlocks"""

    def __init__(self, channels_in: int, channels_out: int, time_emb_dim: int = 0, cond_emb_dim: int = 0, groups: int = 8) -> None:
        """Constructor

        Args:
            channels_in (int): number of channels (filters) in
            channels_out (int): number of channels (filters) out
            time_emb_dim (int): dimension of time embedding (no time embedding if 0)
            cond_emb_dim (int): dimension of class embedding (no time embedding if 0)
            groups (int): number of conv groups

        """
        super().__init__()
        self.emb_dim = int(time_emb_dim) + int(cond_emb_dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.emb_dim, channels_out * 2)) if (time_emb_dim is not None or cond_emb_dim is not None) else None
        self.block1 = ConvBlock(channels_in, channels_out, groups = groups)
        self.block2 = ConvBlock(channels_out, channels_out, groups = groups)
        self.res_conv = nn.Conv1d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None, cond_emb: torch.Tensor = None) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input
            time_emb (torch.Tensor): time embedding
            cond_emb (torch.Tensor): conditioning embedding

        Returns:
            torch.Tensor: output
        """

        scale_shift = None
        if self.mlp is not None and (time_emb is not None or cond_emb is not None):
            cond_emb = tuple(filter(exists, (time_emb, cond_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim = 1)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)



class ConvLinearSelfAttention(nn.Module):
    """Linear self attention with convolution as qkv projections"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 4, head_size: int = 32) -> None:
        """Constructor
        
        Args:
            embedding_dim (int): number of channels (filters) in and out
            num_heads (int): number of heads
            head_size (int): dimension of each head
        """
        super().__init__()
        self.scale = head_size ** -0.5
        self.num_heads = num_heads
        hidden_dim = head_size * num_heads
        self.to_qkv = nn.Conv1d(embedding_dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, embedding_dim, 1), RMSNorm(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.num_heads), qkv)
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.num_heads)
        return self.to_out(out)

class ConvSelfAttention(nn.Module):
    """Self attention with convolution as qkv projections"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 4, head_size: int = 32) -> None:
        """Constructor

        Args:
            embedding_dim (int): number of channels (filters) in and out
            num_heads (int): number of heads
            head_size (int): dimension of each head
        """
        super().__init__()

        self.scale = head_size ** -0.5
        self.num_heads = num_heads
        hidden_dim = head_size * num_heads
        self.to_qkv = nn.Conv1d(embedding_dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input 
        
        Returns:
            torch.Tensor: output
        """

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.num_heads), qkv)
        q = q * self.scale
        sim = einsum(q, k, 'b h d i, b h d j -> b h i j')
        attn = sim.softmax(dim = -1)
        out = einsum(attn, v, 'b h i j, b h d j -> b h i d')
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)