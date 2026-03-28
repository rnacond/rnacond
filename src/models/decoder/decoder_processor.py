from typing import Literal, Optional
from src.dataops.dataloaders.gene_dataset import GeneDataset, GeneDatasetConfig
from src.dataops.dataloaders.data_processor import BaseDataProcessor, DataLoaderConfig
import anndata as ad
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer

class DecoderDatasetConfig(GeneDatasetConfig):
    class Config: extra = 'forbid'
    mask: Literal['causal'] = 'causal' # Masking strategy
    one_sequence: Literal[True] = True # metadata and genes in one sequence (decoder and encoder only) or separately (encoder-decoder and clip)
    
    
class DecoderDataset(GeneDataset):
    def __init__(self, adata: ad.AnnData, config: DecoderDatasetConfig, gene_tokenizer: GeneTokenizer) -> None:
        """Constructor

        Args:
            adata (ad.AnnData): data
            config (DecoderDatasetConfig): config
            gene_tokenizer (Tokenizer): gene tokenizer
        """
        super().__init__(adata, config, gene_tokenizer)
        
class DecoderProcessor(BaseDataProcessor):
    def __init__(self, 
                 dataset_config: DecoderDatasetConfig, 
                 dataloader_config: DataLoaderConfig,
                 adata: Optional[ad.AnnData] = None, 
                 tokenizer: Optional[GeneTokenizer] = None
                 ) -> None:
        """EncoderDataProcessor class. Creates Dataset and DataLoader objects

        Args:
            dataset_config (EncoderDatasetConfig): Dataset configuration
            dataloader_config (DataLoaderConfig): Dataloader configuration
            adata (ad.AnnData): data to define the tokenizer
            tokenizer (Tokenizer): gene tokenizer
        """
        assert tokenizer is not None or adata is not None
        tokenizers = {'gene_tokenizer': tokenizer if tokenizer is not None else GeneTokenizer(adata, dataset_config.metadata_names)}
        super().__init__(DecoderDataset, dataset_config, dataloader_config, tokenizers)
        


