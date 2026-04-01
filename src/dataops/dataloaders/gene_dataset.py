import torch
import anndata as ad
from typing import Union, Literal, Optional, Any
from typing_extensions import Self
import random
from pydantic import model_validator
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer
from src.utils.utils import grouped_shuffle, to_config
from src.dataops.dataloaders.base_dataset import BaseDataset, BaseDatasetConfig
from transformers import DistilBertTokenizer
from torch import Tensor
import pandas as pd
import numpy as np


class GeneDatasetConfig(BaseDatasetConfig):
    shuffle: bool = True # shuffle the genes
    sample: bool = False # sample genes for transformers
    shuffle_metadata: bool = False # shuffle metadata
    num_total_counts: Optional[int] = 1000 # sample this many genes if sample is True
    block_size: Optional[int] = 512 # block size
    metadata_position: Optional[Literal['start', 'end']] = 'start' # metadata position in the sequence
    mask: Literal['none', 'random', 'causal'] = 'none' # Masking strategy
    mask_prob: float = 0.15 # masking probability
    pad: bool = False # pad here or not (in collate function)
    
    
    @model_validator(mode = "after")
    def check_sampling(self) -> Self:
        print("[X] Checking sampling args")
        if self.sample and (self.num_total_counts is None or self.num_total_counts <= 0):
            raise ValueError("num_total_counts must be set and positive when sample is True")
        return self
    
    @model_validator(mode = "after")
    def check_shuffle(self) -> Self:
        print("[X] Checking shuffling metadata args")
        if not self.shuffle_metadata and self.metadata_mask_prob == 0:
            raise ValueError("metadata_mask_prob can't be zero when shuffle_metadata is False")
        return self
    
    
    


class GeneDataset(BaseDataset):
    def __init__(self, adata: ad.AnnData, config: GeneDatasetConfig, gene_tokenizer: GeneTokenizer, text_tokenizer: Optional[DistilBertTokenizer] = None) -> None:
        """Constructor

        Args:
            adata (ad.AnnData): data
            config (GeneDatasetConfig): config
            gene_tokenizer (Tokenizer): gene tokenizer
            text_tokenizer (Optional[DistilBertTokenizer], optional): text tokenizer. Defaults to None.
        """
        super().__init__(adata, config, text_tokenizer)
        
        assert all(x in gene_tokenizer.gene_list for x in self.gene_list), 'adata contains unknown genes'
        self.config = config
        self.gene_tokenizer = gene_tokenizer
        self.block_size = compute_block_size(config)
        self.config.block_size = self.block_size
        self.unknown_token_id = self.gene_tokenizer.special_tokens_ids['unknown_token']
        self.metadata_in_sequence = self.config.one_sequence and self.config.metadata_names is not None and len(self.config.metadata_names) > 0
        self.tokenized_metadata = self.tokenize_dataset_metadata() if self.metadata_in_sequence else None
        self.tokenized_gene_list = self.gene_tokenizer.tokenize(self.gene_list)
            
    def tokenize_dataset_metadata(self) -> pd.Series:
        """Tokenize dataset metadata

        Returns:
            pd.Series: tokenized metadata, each element is a list[int]
        """
        print("[X] Tokenizing metadata")
        return self.adata.obs[self.config.metadata_names].map(self.gene_tokenizer.tokenize).apply(lambda row : [x for x in row], axis = 1)
        
    
    def sample_genes(self, idx: int) -> list[int]:
        """ Returns a sequence of sampled genes
        
        Args:
            idx (int): index
            
        Returns:
            list[int]: sampled tokenized genes
        """
        
        
        counts = self.counts[idx][0].toarray().flatten() if not isinstance(self.counts, np.ndarray) else self.counts[idx].flatten()
        if self.config.sample:
            genes = random.choices(self.tokenized_gene_list, weights = counts, k = self.config.num_total_counts)
        else:
            genes = [gene for i , gene in enumerate(self.tokenized_gene_list) for _ in range(int(counts[i]))]
        return grouped_shuffle(genes, self.config.shuffle, self.tokenized_gene_list) 

    def get_metadata(self, idx: int) -> list[int]:
        """ Returns a row of metadata

        Args:
            idx (int): index

        Returns:
            list[int]: tokenized metadata terms
        """
        
        assert self.tokenized_metadata is not None, 'Metadata is not tokenized. You must set metadata_names in the config'
        obs = [x if random.random() > self.config.metadata_mask_prob else self.unknown_token_id for x in self.tokenized_metadata.iloc[idx]]
        return random.sample(obs, len(obs)) if self.config.shuffle_metadata else obs
    
    
    def create_sequence(self, genes: list[int], metadata: Optional[list[int]] = None) -> list[int]:
        """Assemble genes and metadata and add special tokens

        Args:
            genes (list[int]): genes
            metadata (Optional[list[int]], optional): metadata values. Defaults to None.

        Returns:
            list[int]: sequence of sampled genes and metadata values
        """
        
        if metadata is None or len(metadata) == 0:
            sequence = genes
        elif self.config.metadata_position == 'end':
            sequence = genes + [self.gene_tokenizer.special_tokens_ids['gene_end_token']] + metadata
        elif self.config.metadata_position == 'start':
            sequence = metadata + [self.gene_tokenizer.special_tokens_ids['gene_start_token']] + genes
        return [self.gene_tokenizer.special_tokens_ids['start_token']] + sequence + [self.gene_tokenizer.special_tokens_ids['end_token']] 
    
    
    def random_mask(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """mask the input 

        Args:
            x (Tensor): input

        Returns:
            (Tensor): x masked
            (Tensor): targets
            (Tensor): mask
        """
        mask = (torch.randn(x.size()) < self.config.mask_prob).bool()
        y = x.clone()
        x = x.masked_fill_(mask, self.gene_tokenizer.mask_idx)
        y = y.masked_fill_(~mask, self.gene_tokenizer.mask_idx)
        return x, y, mask
    
    def mask(self, sequence: Tensor) -> dict[str, Tensor]:
        """mask the input sequence

        Args:
            sequence (Tensor): input sequence
            masking_strategy (Literal['causal', 'custom', 'none']): masking strategy

        Returns:
            Dict[str, Tensor]: dictionary with keys 'x', 'y' (causal and random masking) and 'mask' (random masking)
        """
        if self.config.mask == 'causal':
            x, y = sequence[:-1], sequence[1:]
            return {'x': x, 'y': y}
        elif self.config.mask == 'random':
            x, y, mask = self.random_mask(sequence)
            return {'x': x, 'y': y, 'mask': mask}
        elif self.config.mask == 'none':
            return {'x' : sequence}
        else:
            raise ValueError(f"Masking strategy {self.config.mask} not supported")
        
    def random_start_crop(self, sequence: list[Any], target_size: int) -> list[Any]:
        """Returns subsequence of length target_size starting at a random position in sequence

        Args:
            sequence (List[Any]): List to crop
            target_size (int): crop size

        Returns:
            List[Any]: cropped sequence
        """
        if len(sequence) <= target_size:
            return sequence
        random_start = random.randint(0, len(sequence) - target_size)
        return sequence[random_start: random_start + target_size]
        
    def pad(self, tensors: dict[str, Tensor], target_size: int) -> dict[str, Tensor]:
        """Pads one dimensional tensors

        Args:
            tensors (dict[str, Tensor]): one_dimensional tensors to be padded
            target_size (int): target size
            pad_token (str, optional): pad token. Defaults to SPECIAL_TOKENS['pad_token'].

        Returns:
            dict[str, Tensor]: padded tensor
        """
        
        tensors_list = list(tensors.values())
        assert sum([tensor.size() == tensors_list[0].size() for tensor in tensors_list]) == len(tensors), 'All tensors to be padded should have the same size'
        
        size = tensors_list[0].size(-1)
        if size >= target_size:
            return tensors
        else:
            pad_ = lambda x, pad_value, target_size : torch.hstack([x, Tensor([pad_value] * (target_size - x.size(-1)))])
            for key, tensor in tensors.items():
                if key == 'mask':
                    tensors[key] = pad_(tensor, False, target_size).bool()
                else:
                    tensors[key] = pad_(tensor, self.gene_tokenizer.pad_idx, target_size).long()
            tensors['padding_mask'] = torch.hstack([torch.zeros(size), torch.ones(target_size - size)]).bool()
            return tensors
        
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """ Returns a sample from the dataset 

        Args:
            idx (int): index

        Returns:
            dict[str, Tensor]: dictionary with keys 'x', 'y' (causal and random masking), 'mask' (random masking), and maybe 'text_ids' and 'text_mask'
        """

        genes = self.sample_genes(idx)
        metadata, text = (self.get_metadata(idx), None) if self.metadata_in_sequence else (None, self.create_metadata_description(idx))
        sequence = self.create_sequence(genes, metadata)
        target_size = self.block_size + 1 if self.config.mask == 'causal' else self.block_size
        sequence = self.random_start_crop(sequence, target_size)
        sequence = torch.tensor(sequence).long()
        out_genes = self.mask(sequence)
        out_genes = self.pad(out_genes, self.block_size) if self.config.pad else out_genes # padding in the collate function if not here
        out_text = self.tokenize_text(text) if text is not None else {}
        return {**out_genes, **out_text}
    
        
        

def compute_block_size(config: Union[dict[str, Any], GeneDatasetConfig], verbose: bool = False) -> int:
    """Compute block size (max context length)

    Args:
        config (Union[dict[str, Any], GeneDatasetConfig]): config
        verbose (bool, optional): verbose. Defaults to False.

    Raises:
        ValueError: When sample is false block_size is needed

    Returns:
        int: block size that will be used
    """

    config = to_config(config, GeneDatasetConfig)
    if config.sample:
        if config.num_total_counts is None:
            raise ValueError('num_total_counts must be set when sampling')
        block_size = config.num_total_counts + 2 
        if config.one_sequence and config.metadata_names is not None and len(config.metadata_names) > 0:
            block_size = block_size + len(config.metadata_names) + 1
        if config.mask == 'causal':
            block_size = block_size - 1
    else:
        if config.block_size is None:
            raise ValueError('Block size must be set when not sampling')
        block_size = config.block_size
    if verbose:
        print('[X] Since sampling is', config.sample, 'block size set to', block_size, '\n')
    return block_size




