import torch
import anndata as ad
from src.dataops.dataloaders.base_dataset import BaseDataset, BaseDatasetConfig
from typing import Literal
import torchvision
import math
import numpy as np
from transformers import DistilBertTokenizer


class CountsDatasetConfig(BaseDatasetConfig):
    """Configuration for CountsDataset"""
	
    one_sequence: Literal[False] = False
    add_channel_dim: bool = True # add channel dimension
	



class CountsDataset(BaseDataset):
    """Dataset that returns counts and textual metadata"""
	
    def __init__(self, adata: ad.AnnData, config: CountsDatasetConfig, text_tokenizer: DistilBertTokenizer) -> None:
        """Constructor

        Args:
            adata (ad.AnnData): data
            config (CountsDatasetConfig): config
            text_tokenizer (DistilBertTokenizer): text tokenizer
        """
        super().__init__(adata, config, text_tokenizer)
        
        self.seq_length = 16 * math.ceil(self.num_genes / 16)
        print(f'[X] seq length: {self.seq_length}, num genes: {self.num_genes}')
        self.transform = torchvision.transforms.Compose(self.create_transformantions_list())
        self.config = config

    def create_transformantions_list(self) -> list[object]:
        """Create transformations list
		
        Returns:
            List[object]: list of transformations
        """

        transformations = [ToTensor(), Pad(self.seq_length)]
        if self.config.add_channel_dim:
            transformations.append(AddChannelDim())
        return transformations
		

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """ Returns a sample from the dataset 

        Args:
            idx (int): index

        Returns:
            Dict[str, torch.Tensor]: dictionary with keys 'x', 'text_ids', 'text_mask'
        """

        counts = self.counts[idx][0].toarray().flatten()
        counts = self.transform(counts)
        text = self.create_metadata_description(idx)
        out = self.tokenize_text(text)
        out['x'] = counts
        return out
    

    
class MissingToZero(object):
	"""Replace missing values by zeros"""
	
	def __init__(self, missing_value: float = -1.) -> None:
		"""Constructor
		
        Args:
		    missing_value (float): current missing value to be replaced by 0
        """
		
		self.missing_value = missing_value
		
	def __call__(self, sample) -> torch.Tensor:
		"""Call method
		
        Args:
            sample (torch.Tensor): input array
        
        Returns:
            torch.Tensor: output array
        """

		sample[sample == self.missing_value] = 0.
		return sample

class Normalize(object):
	"""Normalize count vector to sum to target_sum"""
	
	def __init__(self, target_sum: int = 10000) -> None:
		"""Constructor
        
		Args:
		    target_sum (int): value to sum to
        """
		
		self.target_sum = target_sum
		
	def __call__(self, sample: torch.Tensor) -> torch.Tensor:
		"""Call method
            
        Args:
            sample (torch.Tensor): input array
            
        Returns:
            torch.Tensor: output array
        """
		sample = self.target_sum * (sample / sample.sum(dim = -1).unsqueeze(-1))
		return sample 

class Log1p(object):
	"""Take the log1p of the counts"""
	
	def __call__(self, sample: torch.Tensor) -> torch.Tensor:
		"""Call method
		
        Args:
            sample (torch.Tensor): input array
        
        Returns:
            torch.Tensor: output array
        """ 
		
		sample = torch.log1p(sample)
		return sample 


class ToTensor(object):
	"""Convert ndarray counts to tensors."""
	
	def __call__(self, sample: np.ndarray) -> torch.Tensor:
		"""Call method
		    
        Args:
            sample (np.ndarray): input array
            
        Returns:
            torch.Tensor: output array
        """ 
		
		sample = torch.from_numpy(sample).type(torch.FloatTensor).squeeze()
		return sample
	

class AddChannelDim(object):
	"""Add channels dimension"""
	
	def __call__(self, sample: torch.Tensor) -> torch.Tensor:
		"""Call method
		
        Args:
            sample (torch.Tensor): input array
        
        Returns:
            torch.Tensor: output array
        """
		
		sample = sample.reshape(1, sample.size(0))
		return sample
	
class Pad(object):
    """Pad to a given length"""
    
    def __init__(self, pad_to: int) -> None:
        """Constructor
        
        Args:
            pad_to (int): length to pad to
        """
        
        self.pad_to = pad_to
        
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Call method
        
        Args:
            sample (torch.Tensor): input array
        
        Returns:
            torch.Tensor: output array
        """
        
        if sample.size(0) < self.pad_to:
            sample = torch.hstack([sample, torch.zeros(self.pad_to - sample.size(0))])
        return sample