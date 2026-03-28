import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Union, Optional, Literal
from typing_extensions import Self
from src.modules.attention.attention import SelfAttentionBlock
from pydantic import BaseModel, model_validator, ConfigDict
from src.utils.utils import sum_masks, initialization
from src.metrics.losses import masked_cross_entropy
from src.modules.encoding import get_position_embedding_layer
from src.utils.constants import PAD_IDX
from src.utils.constants import BERT_SPECIAL_TOKENS_EMBEDDINGS_PATH, BERT_GENE_EMBEDDINGS_PATH, PAD_IDX, BERT_IBD_METADATA_EMBEDDINGS_PATH


class TransformerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True)
    vocab_size: int # Vocabulary size. Get it from the tokenizer or the processor
    block_size: int # Context length. Get it from the processor
    embedding_dim: int = 64 # Embedding dimension = head_size x num_heads
    num_heads: int = 4 # Number of heads
    num_layers: int = 6 # Number of transformer blocks
    dropout: float = 0.2 # Dropout rate
    classification: bool = False # Whether the model is for classification or not
    num_classes: Optional[int] = None # Number of classes for training/finetuning on classification tasks
    causal: bool = False # Masking strategy
    cell_embedding_idx: int = 0 # Which token's embedding is to be considered the cell embedding
    position: Literal['embedding', 'encoding', 'none'] = 'embedding' # Positional encoding strategy
    init_name: Literal['normal', 'xavier'] = 'xavier'
    pad_idx: int = PAD_IDX
    pretrained_embeddings: bool = False # intialize at pretrained embeddings or not
    pretrained_special_tokens_embeddings_path: Optional[str] = BERT_SPECIAL_TOKENS_EMBEDDINGS_PATH
    pretrained_gene_embeddings_path: Optional[str] = BERT_GENE_EMBEDDINGS_PATH
    pretrained_metadata_embeddings: Optional[Union[str, Tensor]] = BERT_IBD_METADATA_EMBEDDINGS_PATH
    fix_embeddings: bool = False # fix the pretrained embeddings or not
    pad_idx: int = PAD_IDX
    
    
    @model_validator(mode = "after")
    def check_classification(self) -> Self:
        print("[X] Checking classification args")
        if self.classification and self.num_classes is None:
            raise ValueError("Num_classes must be set for classification tasks")
        return self
    
    @model_validator(mode = "after")
    def check_dims(self) -> Self:
        print("[X] Checking model dimensions\n")
        if not self.embedding_dim % self.num_heads == 0:
            raise ValueError('Embedding_dim should be divisible by num_heads')
        return self
        
    
class Transformer(nn.Module):
    """Model"""
    def __init__(self, config: TransformerConfig, **kwargs) -> None:
        """Constructor

        Args:
            config (TransformerConfig): model configuration
        """
        super().__init__(**kwargs)
        self.device = None
        self.config = config
        assert config.embedding_dim % config.num_heads == 0, 'embedding_dim should be divisible by num_heads'
        
        if config.pretrained_embeddings:
            special_tokens_embeddings = torch.load(config.pretrained_special_tokens_embeddings_path, weights_only = True)
            gene_embeddings = torch.load(config.pretrained_gene_embeddings_path, weights_only = True)
            metadata_embeddings = torch.load(config.pretrained_metadata_embeddings, weights_only = True) if isinstance(config.pretrained_metadata_embeddings, str) else config.pretrained_metadata_embeddings
            embeddings = torch.vstack((special_tokens_embeddings, gene_embeddings, metadata_embeddings))
            assert config.vocab_size == embeddings.size(0), 'vocab size param and number of embedded tokens must be the same'
            self.token_embedding_table = nn.Embedding.from_pretrained(embeddings, freeze = config.fix_embeddings, padding_idx = config.pad_idx)
            self.token_embedding_projection = nn.Linear(embeddings.size(1), config.embedding_dim) if config.embedding_dim != embeddings.size(1) else nn.Identity()
            
        else:    
            self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx = config.pad_idx)
            self.token_embedding_projection = nn.Identity()
            
        self.position_embedding = get_position_embedding_layer(config.position, config.block_size, config.embedding_dim) 
        self.blocks = nn.ModuleList([SelfAttentionBlock(config.embedding_dim, config.num_heads, config.causal, config.dropout) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embedding_dim) # final layer norm
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes) if config.classification else None
        self.apply(lambda m : initialization(m, init_name = config.init_name, init_embeddings = not config.pretrained_embeddings))


    def get_token_embeddings(self, idx: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Get last embeddings before final linear prediciton head
        
        Args:
            idx (Tensor): input of size (batch, seq_length)
            mask (Optional[Tensor], optional): Mask of size (batch_size, seq_len). Defaults to None.
            padding_mask (Optional[Tensor], optional): Padding mask of size (batch_size, seq_len). Defaults to None.
            
        Returns:
            Tensor: embeddings of size (batch_size, seq_length, embedding_dim)
        """

        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (batch, seq_len1, embedding_dim)
        tok_emb = self.token_embedding_projection(tok_emb) # (batch, seq_len1, embedding_dim)
        pos_emb = self.position_embedding(torch.arange(T).to(idx)) if self.position_embedding is not None else 0
        x = tok_emb + pos_emb # (batch, seq_length, embedding_dim)
        mask = sum_masks(mask, padding_mask)
        for block in self.blocks:
            x = block(x, mask) # (batch, seq_length, embedding_dim)
        x = self.ln_f(x) # (batch, seq_length, embedding_dim)
        return x
    
    def get_cell_embeddings(self, idx: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None, target_token_idx: Union[int, None] = None) -> Tensor:
        """Get last embeddings before final linear prediciton head
        
        Args:
            idx (Tensor): input of size (batch, seq_length)
            mask (Optional[Tensor], optional): Mask of size (batch_size, seq_length). Defaults to None.
            padding_mask (Optional[Tensor], optional): Padding mask of size (batch_size, seq_length). Defaults to None.
            target_token_idx (Union[int, None]): index of the target token whose embedding will be considered the cell embedding
            
        Returns:
            Tensor: embeddings of size (batch_size, embedding_dim)
        """
        target_token_idx = target_token_idx if target_token_idx is not None else self.config.cell_embedding_idx
        x = self.get_token_embeddings(idx, sum_masks(mask, padding_mask))
        return x[:, target_token_idx, :]

    def forward(self, gene_ids: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward function, returns scores over the vocab size

        Args:
            gene_ids (Tensor): input of size (batch, seq_length)
            mask (Optional[Tensor], optional): Mask of size (batch_size, seq_length). Defaults to None.
            padding_mask (Optional[Tensor], optional): Padding mask of size (batch_size, seq_length). Defaults to None.

        Returns:
            Tensor: logit output of size (batch, time, vocab_size)
        """
        x = self.get_token_embeddings(gene_ids, sum_masks(mask, padding_mask))
        scores = self.lm_head(x) # (batch, seq_length, vocab_size)
        return scores
    
    def _loss(self, scores: Tensor, targets: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Calls masked cross entropy loss

        Args:
            scores (Tensor): scores of size (batch_size, seq_len, vocab_size)
            targets (Tensor): targets of size (batch_size, seq_len)
            mask (Optional[Tensor], optional): Mask of size (batch_size, seq_length). Defaults to None.
        
        Returns:
            Tensor: loss
        """
        return masked_cross_entropy(scores, targets, None if self.config.causal else mask)
    
    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Calls forward and then _loss

        Args:
            batch (Dict[str, Tensor]): a batch containing 'x', 'y', 'mask' and 'padding_mask'
        
        Returns:
            Tensor: loss
        """
        idx, targets, mask, padding_mask = batch.get('x'), batch.get('y'), batch.get('mask'), batch.get('padding_mask')
        if targets is None:
            raise ValueError("No targets (key 'y') in batch. Probably because mask is none for an Encoder. Cannot compute loss. Try custom mask.")
        scores = self.forward(idx, sum_masks(mask, padding_mask))
        loss = self._loss(scores, targets, mask)    
        return loss

    def classification_forward(self, idx: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None, targets: Optional[Tensor] = None) -> dict[str, Union[Tensor, float]]:
        """Forward function for classification tasks

        Args:
            idx (Tensor): input of size (B, T)
            mask (Optional[Tensor], optional): Mask. Defaults to None.
            padding_mask (Optional[Tensor], optional): Padding mask. Defaults to None.
            targets (Optional[Tensor], optional): targets of size (B,), if available loss and accuracy will be computed

        Returns:
            Dict[str, Union[Tensor, float]]: output dictionary
        """ 
        if not self.config.classification:
            raise ValueError('Model is not set to classification task')
        else:
            out = dict()
            x = self.get_cell_embeddings(idx, sum_masks(mask, padding_mask))    
            x = x.view(x.size(0), -1)
            out['scores'] = self.classifier(x)
            out['predictions'] = torch.max(out['scores'].data, 1)[1]
            if targets is not None:
                out['loss'] = F.cross_entropy(out['scores'], targets)
                out['accuracy'] = (out['predictions'] == targets).sum().item() / len(targets)
            return out