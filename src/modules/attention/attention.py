"""MHA"""

from torch import nn, Tensor
import torch    
from typing import Optional
import math
from src.modules.linear import FeedFoward
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """MHA"""
    def __init__(self, embedding_dim: int, num_heads: int, causal: bool = False, dropout: float = 0.1, flash: bool = True) -> None:
        """Constructor

        Args:
            embedding_dim (int): embedding dim
            num_heads (int): number of heads. must divide embedding dim
            causal (bool): causal mask or not
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super(MultiHeadAttention, self).__init__()
    
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.causal = causal
        self.flash = flash
               
        self.embedding_dim = embedding_dim 
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads # Dimension of each head's key, query, and value
        
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        
        self.wq = nn.Linear(embedding_dim, embedding_dim) # Query transformation
        self.wk = nn.Linear(embedding_dim, embedding_dim) # Key transformation
        self.wv = nn.Linear(embedding_dim, embedding_dim) # Value transformation
        self.wo = nn.Linear(embedding_dim, embedding_dim) # Output transformation
        
    def split_heads(self, x: Tensor) -> Tensor:
        """Reshape the input to have num_heads for multi-head attention

        Args:
            x (Tensor): input of shape (batch_size, seq_length, embedding_dim)

        Returns:
            Tensor: reshaped input of shape (batch_size, num_heads, seq_length, head_size)
        """

        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
    def combine_heads(self, x: Tensor) -> Tensor:
        """Combine the multiple heads back to original shape

        Args:
            x (Tensor): input of shape (batch_size, num_heads, seq_length, head_size)

        Returns:
            Tensor: reshaped input of shape (batch_size, seq_length, embedding_dim)
        """

        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
    
    def mask_attention_scores(self, attentionn_scores: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Masks values in attention scores

        Args:
            attentionn_scores (Tensor): attention scores of shape (batch_size, num_heads, seq_len1, seq_len2)
            mask (Optional[Tensor]): mask of shape (batch_size, seq_len2)

        Returns:
            Tensor: masked attention scores of shape (batch_size, num_heads, seq_len1, seq_len2)
        """
        batch_size, _, seq_len1, seq_len2 = attentionn_scores.size()
        if self.causal :
            assert seq_len1 == seq_len2, "For causal mask q and kv must have the same length"
            attentionn_scores = attentionn_scores.masked_fill(torch.triu(torch.ones(seq_len1, seq_len1).to(attentionn_scores), diagonal = 1) == 1, float('-inf')) 
        if mask is not None:
            if mask.size() != (batch_size, seq_len2):
                raise ValueError("Mask should have shape (batch_size, kv seq length")
            attentionn_scores = attentionn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) > 0, float('-inf')) 
        return attentionn_scores
        
    def scaled_dot_product_attention(self, 
                                     q: Tensor, 
                                     k: Tensor, 
                                     v: Tensor, 
                                     mask: Optional[Tensor] = None
                                     ) -> Tensor:
        """Attention

        Args:
            q (Tensor): queries of shape (batch_size, num_heads, seq_len1, head_size)
            k (Tensor): keys of shape (batch_size, num_heads, seq_len2, head_size)
            v (Tensor): values of shape (batch_size, num_heads, seq_len2, head_size)
            mask (Tensor, optional): mask of shape (batch_size, seq_len2). Defaults to None.

        Returns:
            Tensor: updated q of shape (batch_size, num_heads, seq_len1, head_size)
        """
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size) # (batch_size, num_heads, seq_len1, seq_len2)
        attention_scores = self.mask_attention_scores(attention_scores, mask)  # (batch_size, num_heads, seq_len1, seq_len2)
        attention_probs = torch.softmax(attention_scores, dim = -1)  # (batch_size, num_heads, seq_len1, seq_len2)
        attention_probs = self.dropout_layer(attention_probs) # (batch_size, num_heads, seq_len1, seq_len2)
        output = torch.matmul(attention_probs, v) # (batch_size, num_heads, seq_len1, head_size)
        return output
    
    def create_attention_mask(self, 
                              batch_size: int,
                              seq_len1: int,
                              seq_len2: int,
                              kv_mask: Optional[Tensor] = None
                              ) -> Optional[Tensor]:
        """Creates mask to be passed on to F.scaled_dot_product_attention

        Args:
            batch_size (int): batch size
            seq_len1 (int): sequence length 1
            seq_len2 (int): sequence length 2
            kv_mask (Optional[Tensor], optional): mask of shape (batch_size, seq_len2). Defaults to None.

        Returns:
            Optional[Tensor]: mask
        """
        
        if not self.causal and kv_mask is None:
            return None
        
        mask = torch.zeros((batch_size, self.num_heads, seq_len1, seq_len2))
        
        if self.causal:
            assert seq_len1 == seq_len2, "For causal mask q and kv must have the same length"
            triu = torch.triu(torch.ones(seq_len1, seq_len1), diagonal = 1) 
            triu = triu.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_heads, 1, 1)
            mask += triu
            
        if kv_mask is not None:
            mask.to(kv_mask)
            mask += kv_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, seq_len1, 1) 
        
        return mask == 0
    
            
        
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward

        Args:
            q (Tensor): queries of shape (batch_size, seq_len1, embedding_dim)
            k (Tensor): keys of shape (batch_size, seq_len2, embedding_dim)
            v (Tensor): values of shape (batch_size, seq_len2, embedding_dim)
            mask (Tensor, optional): mask of shape (batch_size, seq_length2). Defaults to None.

        Returns:
            Tensor: output of shape (batch_size, seq_len, embedding_dim)
        """
        _, seq_len1, _ = q.size()
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        if self.flash:
            create_mask = mask is not None and not self.causal
            mask = ~ (mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, seq_len1, 1) > 0) if create_mask else None 
            a = F.scaled_dot_product_attention(q, k, v, mask, self.dropout, self.causal)
        else:
            a = self.scaled_dot_product_attention(q, k, v, mask) 
        a = self.wo(self.combine_heads(a))
        return a
        
    
    

    
class SelfAttentionBlock(nn.Module):
    """Self attention block that performs self attention followed by a feed forward"""
    def __init__(self, 
                 embedding_dim: int, 
                 num_heads: int, 
                 causal: bool = False, 
                 dropout: Optional[float] = 0.1, 
                 ff_dim: Optional[int] = None
                 ) -> None:
        """Constructor

        Args:
            embedding_dim (int): embedding dimension
            num_heads (int): number of attention heads. Must divide embedding dim
            causal (bool, optional): Causal mask. Defaults to False.
            dropout (Optional[float], optional): Dropout rate. Defaults to 0.1.
            ff_dim (Optional[int], optional): hidden dimension of the feed forward. Defaults to None, in which case it's 4 * embedding_dim
        """
        super(SelfAttentionBlock, self).__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.attention = MultiHeadAttention(embedding_dim, num_heads, causal, dropout)
        self.ff = FeedFoward(embedding_dim, ff_dim, dropout)
        self.lnx = nn.LayerNorm(embedding_dim)
        self.lno = nn.LayerNorm(embedding_dim)
        
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward

        Args:
            x (Tensor): input of shape (batch_size, seq_len, embedding_dim)
            mask (Tensor): mask of shape (batch_size, seq_len)

        Returns:
            Tensor: output of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.lnx(x)
        x = x + self.attention(x, x, x, mask)
        x = x + self.ff(self.lno(x))
        return x
    
    
class CrossAttentionBlock(nn.Module):
    """CrossAttentionBlock that performs cross attention followed by a feed forward"""
    def __init__(self, 
                 embedding_dim: int, 
                 num_heads: int, 
                 ln_kv: bool = True, 
                 dropout: Optional[float] = 0.1, 
                 ff_dim: Optional[int] = None
                 ) -> None:
        """Constructor

        Args:
            embedding_dim (int): embedding dimension
            num_heads (int): number of attention heads. Must divide embedding dim
            ln_kv (bool, optional): whether to apply layer norm to keys and values (not needed in Encoder-Decoder). Defaults to True.
            dropout (Optional[float], optional): Dropout rate. Defaults to 0.1.
            ff_dim (Optional[int], optional): hidden dimension of the feed forward. Defaults to None, in which case it's 4 * embedding_dim
        """
        super(CrossAttentionBlock, self).__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.attention = MultiHeadAttention(embedding_dim, num_heads, False, dropout)
        self.ff = FeedFoward(embedding_dim, ff_dim)
        self.lnx = nn.LayerNorm(embedding_dim)
        self.lnkv = nn.LayerNorm(embedding_dim) if ln_kv else None
        self.lno = nn.LayerNorm(embedding_dim)
        
    def forward(self, q: Tensor, kv: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward

        Args:
            q (Tensor): queries of shape (batch_size, seq_len1, embedding_dim)
            kv (Tensor): keys and values of shape (batch_size, seq_len2, embedding_dim)
            mask (Tensor): kv mask of shape (batch_size, seq_len2)

        Returns:
            Tensor: output of shape (batch_size, seq_len1, embedding_dim)
        """
        q = self.lnx(q)
        kv = self.lnkv(kv) if self.lnkv is not None else kv
        x = q + self.attention(q, kv, kv, mask)
        x = x + self.ff(self.lno(x))
        return x
    
    
class SelfCrossAttentionBlock(nn.Module):
    """Decoder block for an encoder decoder model that performs self attention followed by cross attention"""
    def __init__(self, 
                 embedding_dim: int, 
                 num_heads: int, 
                 causal: bool,
                 ln_kv: bool = False,
                 dropout: float = 0.1, 
                 ff_dim: Optional[int] = None
                 ) -> None:
        """Constructor

        Args:
            embedding_dim (int): embedding dimension
            num_heads (int): number of attention heads. Must divide embedding dim
            causal (bool): Causal mask for self attention. 
            ln_kv (bool, optional): whether to apply layer norm to keys and values (not needed in Encoder-Decoder). Defaults to False.
            dropout (Optional[float], optional): Dropout rate. Defaults to 0.1.
            ff_dim (Optional[int], optional): hidden dimension of the feed forward. Defaults to None, in which case it's 4 * embedding_dim
            
        """
        super(SelfCrossAttentionBlock, self).__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.self_attention = SelfAttentionBlock(embedding_dim, num_heads, causal, dropout, ff_dim)
        self.cross_attention = CrossAttentionBlock(embedding_dim, num_heads, ln_kv, dropout, ff_dim)
        
    def forward(self, 
                q: Tensor, 
                kv: Tensor, 
                q_mask: Optional[Tensor] = None, 
                kv_mask: Optional[Tensor] = None
                ) -> Tensor:
        """Forward

        Args:
            q (Tensor): queries of shape (batch_size, seq_len1, embedding_dim)
            kv (Tensor): keys and values of shape (batch_size, seq_len2, embedding_dim)
            q_mask (Tensor): q mask of shape (batch_size, seq_len1)
            kv_mask (Tensor): kv mask of shape (batch_size, seq_len2)

        Returns:
            Tensor: output of shape (batch_size, seq_len1, embedding_dim)
        """
        q = self.self_attention(q, q_mask)  
        q = self.cross_attention(q, kv, kv_mask)
        return q
    


    



