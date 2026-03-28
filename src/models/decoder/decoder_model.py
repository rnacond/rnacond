
from typing import Union, Literal, Optional
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer
from src.modules.attention.base_decoder_generator import DecoderGenerator
from src.modules.attention.transformer import TransformerConfig, Transformer
from src.modules.base_generative_model import GenerationContext
from torch import Tensor, device


class DecoderConfig(TransformerConfig):
    class Config: 
        extra = 'forbid'
        
    causal: Literal[True] = True # Decoder means causal masking 
    metadata_names: Union[list[str], tuple[str]] = None # Metadata fields in the adata that were extracted to train the model on
    shuffle_metadata: bool = False # 
    num_total_counts: Optional[int] = None # Number of genes that were sampled to train the model on
    sample: bool = True # Whether the genes were sampled or not
    gene_list: list[str]
    gene_symbols: Optional[list[str]] = None
    
    
        
class Decoder(DecoderGenerator, Transformer):
    """Decoder model"""
    def __init__(self, config: DecoderConfig, tokenizer: Optional[GeneTokenizer] = None) -> None:
        """Constructor

        Args:
            config (DecoderConfig): model configuration
            tokenizer (Optional[GeneTokenizer], optional): tokenizer. Defaults to None.
        """
        
        super().__init__(config.gene_list, config.gene_symbols, tokenizer, config.block_size, config.metadata_names, config.sample, config.num_total_counts, config = config)
        self.name = 'Decoder'
        self.get_scores = self.forward
        self.vocab = tokenizer.get_tokens()
        self.config = config
        
        
    def encode_context(self, context: GenerationContext, batch_size, run_on: Union[device, Tensor]) -> dict[str, Tensor]:
        """Encodes raw context for generation

        Args:
            context (GenerationContext): contains 'gene_context' and 'text_context'
            batch_size (int): batch size
            run_on (Union[torch.device, Tensor]): device to run on

        Returns:
            dict[str, Tensor]: contains 'gene_ids', input to get scores method (same as forward of Transformer)
        """

        run_on = run_on if run_on is not None else self.device
        gene_context = context.get_gene_context(self.gene_tokenizer, self.start_token, True, batch_size, run_on, self.metadata_names, self.unknown_token, self.gene_start_token, self.config.shuffle_metadata)  
        return {'gene_ids' : gene_context}
        
