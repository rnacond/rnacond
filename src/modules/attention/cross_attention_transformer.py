import torch
import torch.nn as nn
from typing import Optional, Literal
from src.modules.attention.attention import SelfCrossAttentionBlock
from src.utils.utils import sum_masks, initialization
from src.modules.encoding import get_position_embedding_layer
from torch import Tensor
from pydantic import BaseModel
from src.utils.constants import BERT_SPECIAL_TOKENS_EMBEDDINGS_PATH, BERT_GENE_EMBEDDINGS_PATH, PAD_IDX

class CrossAttentionTransformerConfig(BaseModel):
    vocab_size: int # Vocabulary size. Get it from the tokenizer or the processor
    block_size: int # Context length. Get it from the processor
    embedding_dim: int = 64 # Embedding dimension = head_size x num_heads
    num_heads: int = 4 # Number of heads
    num_layers: int = 6 # Number of transformer blocks
    dropout: float = 0.2 # Dropout rate
    position: Literal['embedding', 'encoding', 'none'] = 'embedding' # Positional encoding strategy
    causal: bool = True # Masking strategy
    ln_kv: bool = False # whether to use layer norm for key-value
    cell_embedding_idx: int = 0 # Which token's embedding is to be considered the cell embedding
    init_name: Literal['normal', 'xavier'] = 'xavier'
    pretrained_embeddings: bool = False # intialize at pretrained embeddings or not
    pretrained_special_tokens_embeddings_path: str = BERT_SPECIAL_TOKENS_EMBEDDINGS_PATH
    pretrained_gene_embeddings_path: str = BERT_GENE_EMBEDDINGS_PATH
    fix_embeddings: bool = False # fix the pretrained embeddings or not
    pad_idx: int = PAD_IDX
    
    
class CrossAttentionTransformer(nn.Module):
    """Cross attention decoder model, the decoder part in an encoder-decoder model"""
    def __init__(self, config: CrossAttentionTransformerConfig) -> None:
        """Constructor

        Args:
            config (CrossAttentionTransformerConfig): config
        """
        super().__init__()
        
        self.device = None
        assert config.embedding_dim % config.num_heads == 0, 'CrossAttentionTransformer embedding_dim should be divisible by num_heads'
        
        if config.pretrained_embeddings:
            special_tokens_embeddings = torch.load(config.pretrained_special_tokens_embeddings_path, weights_only = True)
            gene_embeddings = torch.load(config.pretrained_gene_embeddings_path, weights_only = True)
            embeddings = torch.vstack((special_tokens_embeddings, gene_embeddings))
            assert config.vocab_size == embeddings.size(0), 'vocab size param and number of embedded tokens must be the same'
            self.token_embedding_table = nn.Embedding.from_pretrained(embeddings, freeze = config.fix_embeddings, padding_idx = config.pad_idx)
            self.token_embedding_projection = nn.Linear(embeddings.size(1), config.embedding_dim) if config.embedding_dim != embeddings.size(1) else nn.Identity()
        else:    
            self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx = config.pad_idx)
            self.token_embedding_projection = nn.Identity()
            
        self.position_embedding = get_position_embedding_layer(config.position, config.block_size, config.embedding_dim) 
        self.blocks = nn.ModuleList([SelfCrossAttentionBlock(config.embedding_dim, config.num_heads, config.causal, config.ln_kv, config.dropout) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embedding_dim) # final layer norm
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.apply(lambda m : initialization(m, init_name = config.init_name, init_embeddings = not config.pretrained_embeddings))
        
    def get_token_embeddings(self, idx: Tensor, kv: Tensor, q_mask: Optional[Tensor] = None, q_padding_mask: Optional[Tensor] = None, kv_mask: Optional[Tensor] = None, kv_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Get last embeddings before final linear prediciton head

        Args:
            idx (Tensor): input of shape (batch_size, seq_len1), so not embedded yet
            kv (Tensor): key-value of shape (batch_size, seq_len2, embedding_dim), so already embedded
            q_mask (Tensor, optional): query mask of shape (batch_size, seq_len1)
            q_padding_mask (Tensor, optional): query padding mask of shape (batch_size, seq_len1)
            kv_mask (Tensor, optional): key-value mask of shape (batch_size, seq_len2)
            kv_padding_mask (Tensor, optional): key-value padding mask of shape (batch_size, seq_len2)

        Returns:
            Tuple[Tensor, Tensor]: logits, hidden states
        """
        tok_emb = self.token_embedding_table(idx) # (batch, seq_len1, embedding_dim or pretrained_embedding_dim)
        tok_emb = self.token_embedding_projection(tok_emb) # (batch, seq_len1, embedding_dim)
        pos_emb = self.position_embedding(torch.arange(idx.size(-1)).to(idx)) if self.position_embedding is not None else 0 
        q = tok_emb + pos_emb
        q_mask = sum_masks(q_mask, q_padding_mask)
        kv_mask = sum_masks(kv_mask, kv_padding_mask)
        for block in self.blocks:
            q = block(q, kv, q_mask, kv_mask)
        q = self.ln_f(q)
        return q
    
    def forward(self, idx: Tensor, kv: Tensor, q_mask: Optional[Tensor] = None, q_padding_mask: Optional[Tensor] = None, kv_mask: Optional[Tensor] = None, kv_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass

        Args:
            idx (Tensor): input of shape (batch_size, seq_len1), so not embedded yet
            kv (Tensor): key-value of shape (batch_size, seq_len2, embedding_dim), so already embedded
            q_mask (Tensor, optional): query mask of shape (batch_size, seq_len1)
            q_padding_mask (Tensor, optional): query padding mask of shape (batch_size, seq_len1)
            kv_mask (Tensor, optional): key-value mask of shape (batch_size, seq_len2)
            kv_padding_mask (Tensor, optional): key-value padding mask of shape (batch_size, seq_len2)

        Returns:
            Tensor: logits
        """
        return self.lm_head(self.get_token_embeddings(idx, kv, q_mask, q_padding_mask, kv_mask, kv_padding_mask))