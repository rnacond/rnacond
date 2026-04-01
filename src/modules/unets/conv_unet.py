from functools import partial
import torch
from torch import nn
from einops import repeat
from src.utils.utils import prob_mask_like, initialization
from pydantic import BaseModel
from typing import Optional, Tuple
from collections import namedtuple
from src.modules.conv import Downsample, Upsample, ConvLinearSelfAttention, ConvSelfAttention, CondTimeResidualConvBlock
from src.modules.encoding import SinusoidalPosEnc, RandomOrLearnedSinusoidalPosEnc
from src.modules.layer_utils import PreRMSNorm, Residual
from src.utils.constants import BERT_EMBEDDING_DIM

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class ConvUNetConfig(BaseModel):
    dim: int = 64 # latent dimension
    cond_embedding_dim: int = BERT_EMBEDDING_DIM # conditional embedding dimension
    cond_drop_prob: float = 0.5 # probability of dropping the conditional embedding
    attn_dim_head: int = 32 # attention head dimension
    attn_heads: int = 4 # number of attention heads
    init_dim: Optional[int] = None
    out_dim: Optional[int] = None
    dim_mults: Tuple[int] = (1, 2, 4, 8)
    channels: int = 1
    self_cond: bool = False
    resnet_block_groups: int = 8
    learned_variance: bool = False
    learned_sinusoidal_cond: bool = False
    random_fourier_features: bool = False
    learned_sinusoidal_dim: int = 16
    sinusoidal_pos_emb_theta: int = 10000


class ConvUNet(nn.Module):
    """Conditional UNet denoiser model."""
    def __init__(self, config: ConvUNetConfig) -> None:
        """Constructor
        
        Args:
            config (UNetConfig): configuration
        """
        
        super().__init__()

        self.config = config
        input_channels = config.channels * (2 if config.self_cond else 1)
        init_dim = config.dim if config.init_dim is None else config.init_dim
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        dims = [init_dim, *map(lambda m: config.dim * m, config.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(CondTimeResidualConvBlock, groups = config.resnet_block_groups)
        updown_attention = ConvLinearSelfAttention
        mid_attention = ConvSelfAttention

        # time embedding
        time_dim = config.dim * 4
        self.random_or_learned_sinusoidal_cond = config.learned_sinusoidal_cond or config.random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEnc(config.learned_sinusoidal_dim, config.random_fourier_features)
            fourier_dim = config.learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEnc(config.dim, theta = config.sinusoidal_pos_emb_theta)
            fourier_dim = config.dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(),nn.Linear(time_dim, time_dim))

        # metadata embedding
        text_dim = config.dim * 4
        self.text_emb = nn.BatchNorm1d(config.cond_embedding_dim)
        self.null_text_emb = nn.Parameter(torch.randn(config.cond_embedding_dim))
        self.text_mlp = nn.Sequential(nn.Linear(config.cond_embedding_dim, text_dim), nn.GELU(), nn.BatchNorm1d(text_dim), nn.Linear(text_dim, text_dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = text_dim), 
                                             block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = text_dim),
                                             Residual(PreRMSNorm(dim_in, updown_attention(dim_in))),
                                             Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)]))
            
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = text_dim)
        self.mid_attn = Residual(PreRMSNorm(mid_dim, mid_attention(mid_dim, head_size = config.attn_dim_head, num_heads = config.attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = text_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = text_dim), 
                                           block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = text_dim),
                                           Residual(PreRMSNorm(dim_out, updown_attention(dim_out))), 
                                           Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1)]))
            
        default_out_dim = config.channels * (1 if not config.learned_variance else 2)
        self.out_dim = default_out_dim if config.out_dim is None else config.out_dim
        self.final_res_block = block_klass(config.dim * 2, config.dim, time_emb_dim = time_dim, cond_emb_dim = text_dim)
        self.final_conv = nn.Conv1d(config.dim, self.out_dim, 1)
        self.apply(initialization)



    def forward_with_cond_scale(self, x: torch.Tensor, time: torch.Tensor, text_emb: torch.Tensor, cond_scale: float = 1., rescaled_phi: float = 0.) -> torch.Tensor: 
        """Scaled output between conditining and no conditioning
        
        Args:
            x (torch.Tensor): input tensor
            time (torch.Tensor): time tensor
            text_emb (torch.Tensor): text embedding tensor
            cond_scale (float): scaling factor for the conditioning
            rescaled_phi (float): rescaling factor for the conditioning

        Returns:
            torch.Tensor: output tensor
        """

        logits = self.forward(x, time, text_emb, cond_drop_prob = 0.)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(x, time, text_emb, cond_drop_prob = 1.)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale
        if rescaled_phi == 0.:
            return scaled_logits
        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def drop_cond(self, text_emb: torch.Tensor, cond_drop_prob: float, batch_size: int, null_emb: torch.Tensor) -> torch.Tensor:
        """"Drops the conditional embedding with a given probability, replaces it with null_emb
        
        Args:
            text_emb (torch.Tensor): text embedding tensor
            cond_drop_prob (float): probability of dropping the conditional embedding
            batch_size (int): batch size
            null_emb (torch.Tensor): null embedding tensor
        
        Returns:
            torch.Tensor: conditional embedding tensor
        """
        
        keep_mask = prob_mask_like((batch_size, ), 1 - cond_drop_prob).to(text_emb).bool().unsqueeze(1)
        null_emb = repeat(null_emb, 'd -> b d', b = batch_size) 
        class_emb = torch.where(keep_mask.bool(), text_emb, null_emb)
        return class_emb

    def forward(self, x: torch.Tensor, time: torch.Tensor, text_emb: torch.Tensor, cond_drop_prob: Optional[float] = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            x (torch.Tensor): input tensor
            time (torch.Tensor): time tensor
            text_emb (torch.Tensor): text embedding tensor
            cond_drop_prob (Optional[float]): probability of dropping the conditional embedding

        Returns:
            torch.Tensor: output tensor
        """
        
        cond_drop_prob = self.config.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        batch_size = x.shape[0]
        text_emb = self.text_emb(text_emb)
        if cond_drop_prob > 0:
            text_emb = self.drop_cond(text_emb, cond_drop_prob, batch_size, self.null_text_emb)
        text_emb = self.text_mlp(text_emb)
        
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, text_emb)
            h.append(x)
            x = block2(x, t, text_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t, text_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, text_emb)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, text_emb)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, text_emb)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t, text_emb)
        return self.final_conv(x)
    





