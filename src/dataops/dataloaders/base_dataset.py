"""Base class for datasets"""

import itertools
import torch
import random
from typing import Union
import anndata as ad    
from transformers import DistilBertTokenizer
from typing import Optional, Tuple
from torch.utils.data import Dataset
from typing_extensions import Self
from pydantic import BaseModel, model_validator
from src.utils.utils import create_metadata_sentences, metadata_sentence
import pandas as pd
import numpy as np


class BaseDatasetConfig(BaseModel):
    
    metadata_names: Union[list[str], Tuple[str]] = [] # metadata fields in the adata to consider
    metadata_mask_prob: float = 0.1 # probability of masking metadata
    one_sequence: bool = True # metadata and genes in one sequence (decoder and encoder only) or separately (encoder-decoder and clip)
    max_text_length: Optional[int] = 200 # max text length
    
    @model_validator(mode = "after")
    def check_text_tokenizer(self) -> Self:
        print("[X] Checking text tokenizer args")
        if not self.one_sequence and (self.max_text_length is None or len(self.metadata_names) == 0):
            raise ValueError("max_text_length and metadata_names are needed when one_sequence is False")
        return self


class BaseDataset(Dataset):
    "Contains methods for constructing and tokenizing textual inputs"

    def __init__(self, 
                 adata: ad.AnnData, 
                 config: BaseDatasetConfig,
                 text_tokenizer: Optional[DistilBertTokenizer] = None
                 ) -> None:
        
        """Constructor

        Args:
            adata (ad.AnnData): data
            config (BaseDatasetConfig): config
            text_tokenizer (Optional[DistilBertTokenizer], optional): text tokenizer. Defaults to None. Required if config.one_sequence is False
        """
        
        assert not (not config.one_sequence and text_tokenizer is None), "text_tokenizer is required when one_sequence is False"
        
        
        self.adata = adata
        self.counts = adata.X.toarray()
        self.gene_list = adata.var_names.tolist()
        self.num_genes = len(self.gene_list)
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.metadata_as_text = not self.config.one_sequence and self.config.metadata_names is not None and len(self.config.metadata_names) > 0
        self.tokenized_metadata_text = self.tokenize_dataset_metadata_text() if self.metadata_as_text else None
        
        if 'log1p_total_counts' in adata.obs.columns:
            self.total_counts = list((map(int, np.round(np.exp(adata.obs['log1p_total_counts'].to_numpy()) - 1).tolist())))
        else:
            self.total_counts = list(map(int, self.counts.toarray().sum(axis = 1)))
        
        
    def __len__(self) -> int:
        """Number of samples

        Returns:
            int: number of samples
        """
        return len(self.adata)
		
    def create_metadata_description(self, idx: int) -> str:
        """Creates a sentence from the metadata

        Args:
            idx (int): index

        Returns:
            str: text
        """

        metadata_names = random.sample(self.config.metadata_names, len(self.config.metadata_names))
        metadata_dict = {metadata_name: self.adata.obs[metadata_name].iloc[idx] for metadata_name in metadata_names}
        text = create_metadata_sentences(metadata_dict, self.config.metadata_mask_prob)
        return text
    
    def tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize text

        Args:
            text (str): text

        Returns:
            Dict[str, torch.Tensor]: dictionary with keys 'text_ids', 'text_mask'
        """
        
        out = dict()
        text = self.text_tokenizer(text, padding = 'max_length', truncation = True, max_length = self.config.max_text_length)
        out['text_ids'] = torch.tensor(text['input_ids']).long()
        out['text_mask'] = torch.tensor(text['attention_mask']).long()
        return out
    
        
    def tokenize_dataset_metadata_text(self) -> pd.Series:
        """Creates sentences describing the metadata in the dataset and tokenizes them.
        
        Returns:
            pd.Series: tokenized text metadata, each element is list[list[int]]
        """
        
        print('[X] Tokenizing dataset metadata text')
        self.pad_token_id = self.text_tokenizer.pad_token_id 
        self.delimiter = ','
        self.delimiter_id = self.text_tokenizer(self.delimiter, add_special_tokens = False)['input_ids']
        assert len(self.delimiter_id) == 1
        self.delimiter_id = self.delimiter_id[0]  
        sentence = lambda row: (self.delimiter + ' ').join([metadata_sentence(metadata_name, row[metadata_name]) for metadata_name in self.config.metadata_names])
        tokenize = lambda x : self.text_tokenizer(x, padding = False, truncation = False, return_attention_mask = False)['input_ids']
        split_list = lambda lst, delimiter : [list(y) for x, y in itertools.groupby(lst, lambda z: z == delimiter) if not x]    
        return self.adata.obs[self.config.metadata_names].apply(sentence, axis = 1).map(tokenize).map(lambda x : split_list(x, self.delimiter_id))
    
    def mask_metadata_text(self, lst: list[list[int]]) -> list[int]:
        """Masks metadata text

        Args:
            lst (list[list[int]]): tokenized text

        Returns:
            list[int]: masked tokenized text
        """
        
        lst = [x for x in lst if random.random() > self.config.metadata_mask_prob]  
        join = lambda lst, sep : [number for sublist in lst for number in sublist + [sep]][:-1]
        lst = join(lst, self.delimiter_id)
        lst = [self.text_tokenizer.cls_token_id] + lst if lst[0] != self.text_tokenizer.cls_token_id else lst
        lst = lst + [self.text_tokenizer.sep_token_id] if lst[-1] != self.text_tokenizer.sep_token_id else lst
        return lst
    
    
  