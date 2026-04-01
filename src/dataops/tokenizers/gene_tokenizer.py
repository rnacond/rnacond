
import anndata as ad
from typing import Union, Optional
from src.utils.constants import SPECIAL_TOKENS, GENE_LIST_PATH
from src.dataops.tokenizers.simple_tokenizer import SimpleTokenizer


class GeneTokenizer(SimpleTokenizer):
    """Tokenizer for a sequence made of metadata fields and gene names"""
    def __init__(self, 
                 adata: Optional[ad.AnnData] = None, 
                 metadata_names: Union[list[str], tuple[str]] = [], 
                 gene_list_path: str = GENE_LIST_PATH,
                 special_tokens: dict[str, str] = SPECIAL_TOKENS,
                 gene_list_from_adata: bool = True
                 ) -> None:
        """Constructor

        Args:
            adata (Optional[ad.AnnData]): data to extract metadata terms and genes (if gene_list_from_adata is True) from.
            metadata_names (Union[list[str], tuple[str]): metadata names. 
            gene_list_path (str): path to gene list file. Defaults to GENE_LIST_PATH. Will be used if gene_list is None.
            special_tokens (dict[str, str], optional): special tokens. Defaults to SPECIAL_TOKENS.
            gene_list_from_adata (bool): gene list from adata var names if True, from gene_list_path if True
        """
        
        assert not (adata is None and len(metadata_names) > 0), "metadata_names provided but adata is None, tokenization impossible"
        
        if gene_list_from_adata:
            assert adata is not None, "adata cannot be None if gene_list_from_adata is True"
            self.gene_list = adata.var_names.to_list()
        else:
            with open(gene_list_path) as f:
                self.gene_list = f.read().splitlines()
                if adata is not None and not all(x in self.gene_list for x in adata.var_names.to_list()):
                    raise ValueError('adata contains unknown genes')
        
        self.metadata_names = metadata_names
        self.set_metadata_tokens(adata)
        self.special_tokens = special_tokens
        self.gene_order = { gene : i for i, gene in enumerate(self.gene_list) }
        self.tokens = list(self.special_tokens.values()) + self.gene_list + self.metadata_tokens
        super().__init__(self.tokens)
        self.special_tokens_ids = { key : self.stoi[value] for key, value in self.special_tokens.items() }
        self.tokenized_gene_list = self.tokenize(self.gene_list)
        self.mask_idx = self.special_tokens_ids['mask_token']
        self.pad_idx = self.special_tokens_ids['pad_token']
        
        

    
    def set_metadata_tokens(self, adata: ad.AnnData) -> None:
        """Finds and stores all unique values in the metadata fields
        
        Args:
            adata (ad.AnnData): data

        Returns:
            None
        """
        self.metadata_tokens = []
        self.metadata_tokens_per_column = dict()
        if adata is not None and self.metadata_names is not None and len(self.metadata_names) > 0:
            for metadata_name in self.metadata_names:
                tokens = adata.obs[metadata_name].unique().tolist()
                self.metadata_tokens_per_column[metadata_name] = tokens
                self.metadata_tokens.extend(tokens)
        self.metadata_tokens = list(set(self.metadata_tokens))
        
    
    def update_from_adata(self, adata: ad.AnnData, metadata_names: list[str]= []) -> None:
        """Adds genes and metadata terms to tokenizer

        Args:
            adata (ad.AnnData): adata
            metadata_names (list[str]): metadata columns to consider
        """
        
        new_metadata_names = [metadata_name for metadata_name in metadata_names if metadata_name not in self.metadata_names]
        self.metadata_names += new_metadata_names
        new_metadata_tokens = [metadata_token for metadata_token in self.get_metadata_tokens(adata) if metadata_token not in self.metadata_tokens]
        self.metadata_tokens += new_metadata_tokens
        self.tokens += new_metadata_tokens
        super().__init__(self.tokens)