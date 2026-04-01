
from src.utils.utils import to_config
import anndata as ad
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Type, Union, Literal, Any, Optional, Iterator
from pydantic import BaseModel
from src.dataops.data_utils import pad_and_collate  
from src.dataops.dataloaders.gene_dataset import GeneDataset, GeneDatasetConfig
from src.dataops.data_utils import train_val_split
from src.dataops.dataloaders.counts_dataset import CountsDataset, CountsDatasetConfig 
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer
from transformers import DistilBertTokenizer
from torch import Tensor
import random


class DataLoaderConfig(BaseModel):
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    drop_last: bool = True
    collate_function: Literal['default', 'pad'] = 'pad' # pad for transformers and default for diffusion
    pad_idx: Optional[int] = None
    sampler: bool = False

    
class BaseDataProcessor():
    def __init__(self, 
                 dataset_class: Type[Union[GeneDataset, CountsDataset]], 
                 dataset_config: Union[GeneDatasetConfig, CountsDatasetConfig], 
                 dataloader_config: Union[dict[str, Any], DataLoaderConfig], 
                 tokenizers: dict[str, Union[GeneTokenizer, DistilBertTokenizer]] = {}
                 ) -> None:
        
        """Constructor

        Args:
            dataset_class (Type[BaseDataset]): The Dataset class
            dataset_config (Union[GeneDatasetConfig, CountsDatasetConfig]): The config for the dataset class
            dataloader_config (Union[dict[str, Any], DataLoaderConfig]): Dataloader configuration
            tokenizers (dict[str, Union[GeneTokenizer, DistilBertTokenizer]]): The tokenizers, contains keys gene_tokenizer and/or text_tokenizer, as accepted by dataset_class
        """
        
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config
        dataloader_config = to_config(dataloader_config, DataLoaderConfig)
        self.dataloader_config = dataloader_config.model_copy(update = {'pad_idx' : tokenizers['gene_tokenizer'].special_tokens_ids['pad_token']}) if 'gene_tokenizer' in tokenizers else dataloader_config
        self.tokenizers = tokenizers
        
        if self.dataloader_config.collate_function == 'default':
            self.collate_function = None
        elif self.dataloader_config.collate_function == 'pad':
            assert self.dataloader_config.pad_idx is not None, 'pad_idx must be set when collate_function is pad'
            self.collate_function = self.pad_and_collate_
            
    def pad_and_collate_(self, x: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """"Pads and collates the data using the pad_and_collate function with the given pad token id
        
        Args:
            x (list[dict[str, Tensor]]): list of tensors to pad and collate
            
        Returns:
            dict[str, Tensor]: padded and collated data
        """
        return pad_and_collate(x, self.dataloader_config.pad_idx)
    
    def transform_to_dataset(self, adata: ad.AnnData) -> Union[GeneDataset, CountsDataset]:
        """ Transforms anndata and return torch dataset

        Args:
            adata (ad.AnnData): data

        Returns:
            Union[GeneDataset, CountsDataset]: dataset
        """
        
        dataset = self.dataset_class(adata, self.dataset_config, **self.tokenizers)
        return dataset

    def transform_to_dataloader(self, 
                                data: Union[GeneDataset, CountsDataset, ad.AnnData], 
                                shuffle: bool = True, 
                                *args
                                ) -> Union[DataLoader, tuple[DataLoader, list[Any]]]:
        """ Transforms anndata and return torch dataloader

        Args:
            data (Union[GeneDataset, CountsDataset, ad.AnnData]): data
            shuffle (bool) : shuffle or not 
            *args: other attributes from dataset to return

        Returns:
            Union[DataLoader, tuple[DataLoader, list[Any]]]: dataloader or dataloader and other attributes in a list
        """
        
        if isinstance(data, ad.AnnData):
            dataset = self.transform_to_dataset(data)
        elif isinstance(data, Dataset):
            dataset = data
        else:
            raise TypeError('Data should be either an AnnDataset or a Dataset')
            
        if self.dataloader_config.sampler:
            sampler = LengthBatchSampler(dataset.total_counts, self.dataloader_config.batch_size) 
            dataloader_config = self.dataloader_config
        else:
            sampler = None
            dataloader_config = self.dataloader_config.model_copy(update = {'shuffle' : shuffle}) 
        remove_from_dataloader_config = ['sampler', 'collate_function', 'pad_idx']
        remove_from_dataloader_config = remove_from_dataloader_config + ['shuffle', 'drop_last', 'batch_size'] if self.dataloader_config.sampler else remove_from_dataloader_config
        dataloader_config = {key : value for key, value in dataloader_config.model_dump().items() if key not in remove_from_dataloader_config}
        dataloader = DataLoader(dataset, batch_sampler = sampler, collate_fn = self.collate_function, **dataloader_config)
        return dataloader if len(args) == 0 else (dataloader, [getattr(dataset, key, None) for key in args])
        
    
    


def create_dataloaders(data_processor: BaseDataProcessor,
                       adata: ad.AnnData, 
                       *args
                       ) -> tuple[Any, ...]:
    
    """Creates dataloaders

    Args:
        data_processer (BaseDataProcessor): data_processor
        adata (ad.AnnData): data
        *args: other attributes from dataset to return, args is a list of strings
        
    Returns:
        tuple[Any, ...]: contains train dataloader (DataLoader), valid dataloader (DataLoader), other attributes
    """
    
    train_adata, valid_adata = train_val_split(adata) 
    train_dataloader = data_processor.transform_to_dataloader(train_adata)  
    valid_dataloader, extra = data_processor.transform_to_dataloader(valid_adata, False, *args) if len(args) > 0 else (data_processor.transform_to_dataloader(valid_adata, False), [])

    return train_dataloader, valid_dataloader, *extra


class LengthBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], batch_size: int) -> None:
        """Constructor
        
        Args:
            lengths (list[int]): list of sequence lengths in the dataset
            batch_size (int): batch_size
        """
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.size = len(lengths)
        self.step = 100 * batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """"""
        indices = list(range(self.size))
        random.shuffle(indices)
        
        for i in range(0, self.size, self.step):
            pool = indices[i : i + self.step]
            pool = sorted(pool, key = lambda x : self.lengths[x])
            for j in range(0, len(pool), self.batch_size):
                if j + self.batch_size > len(pool): # assume drop_last = True
                    break
                z = pool[j: j + self.batch_size]
                random.shuffle(z)
                yield z

    def __len__(self) -> int:
        """Returns the numner of batches
        
        Returns:
            int: number of batches"""
        return len(self.lengths) // self.batch_size