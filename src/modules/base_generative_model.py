
from typing import Optional, Union
from transformers import DistilBertTokenizer
import torch
import numpy as np
from src.utils.utils import create_metadata_sentences
from torch import nn, Tensor, device
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer
import anndata as ad
from scipy.sparse import csc_matrix
import pandas as pd


class GenerationContext():
    "Transforms generation context as given by the user to models' input"
    
    def __init__(self, 
                 gene_context: list[str] = [], 
                 metadata_context: dict[str, str] = dict()
                 ) -> None:
        
        """Constructor

        Args:
            gene_context (list[str]): gene context
            metadata_context  (dict[str, str]): metadata context
        """
        
        assert isinstance(gene_context, list) and sum([isinstance(x, str) for x in gene_context]) == len(gene_context), "gene_context must be list of str"
        assert isinstance(metadata_context, dict) and sum([isinstance(k, str) and isinstance(v, str) for k, v in metadata_context.items()]) == len(metadata_context), 'metadata_context must be dict[str, str]'
        
        self.gene_context = gene_context
        self.metadata_context = metadata_context

    def get_text_context(self, 
                         tokenizer: DistilBertTokenizer, 
                         batch_size: int = 1, 
                         run_on: Optional[Union[device, Tensor]] = None
                         ) -> Tensor:
        
        """ Metadata context as tokenized text

        Args:
            tokenizer(DistilBertTokenizer): tokenizer
            batch_size (int): batch_size
            run_on (Optional[Union[torch.device, torch.Tensor]]): device

        Returns:
            torch.Tensor: tokenized text context
        """
        
        text_context = create_metadata_sentences(self.metadata_context)
        text_context = torch.tensor(tokenizer(text_context)['input_ids']).unsqueeze(0).repeat(batch_size, 1).long()
        
        return text_context if run_on is None else text_context.to(run_on)
    
    def get_gene_context(self, 
                         tokenizer: GeneTokenizer, 
                         start_token: str,
                         include_metadata: bool, 
                         batch_size: int = 1, 
                         run_on: Optional[Union[device, Tensor]] = None,
                         metadata_names: Optional[list[str]] = None,
                         unknown_token: Optional[str] = None,
                         gene_start_token: Optional[str] = None,
                         random_order_metadata: bool = False,
                         ) -> Tensor:
        
        """ Gene context tokenized

        Args:
            tokenizer (Tokenizer): tokenizer
            start_token (str): start token
            include_metadata (bool): include metadata terms in the same sequence (for Decoder model)
            batch_size (int): batch_size
            run_on (Optional[Union[torch.device, torch.Tensor]]): device
            metadata_names (Optional[list[str]]): metadata_names model was trained on; necessary if include_metadata is True
            unknown_token (Optional[str]): necessary if include_metadata is True
            gene_start_token (str): necessary if include_metadata i True
            random_order_metadata (bool): whether to include metadata in random order

        Returns
            torch.Tensor: tokenized gene context
        """

        if include_metadata:
            
            unseen_metadata = [key for key in self.metadata_context.keys() if key not in metadata_names]
            if len(unseen_metadata) > 0:
                print('[!] Warning: metadata fields in the context not seen in training:', unseen_metadata)
                
            if random_order_metadata:
                
                metadata_values = [value for key, value in self.metadata_context.items() if key in metadata_names] 
                
                if len(metadata_values) < len(metadata_names):
                    if len(self.gene_context) > 0:
                        print('[!] Warning: Gene context will be ignored because metadata is not complete, new generate function needed to fill holes')
                    gene_context = [start_token] + metadata_values 
                else:
                    gene_context = [start_token] + metadata_values + [gene_start_token] + self.gene_context
                    
            else:
                
                metadata_values = [self.metadata_context.get(key, unknown_token) for key in metadata_names]
                gene_context = [start_token] + metadata_values + [gene_start_token] + self.gene_context
                
        else: 
            gene_context = [start_token] + self.gene_context
            
        gene_context = torch.tensor(tokenizer.tokenize(gene_context)).long().unsqueeze(0).repeat(batch_size, 1)
        return gene_context if run_on is None else gene_context.to(run_on)
    
    def to_str(self) -> str:
        """Returs a string represeting the metadata context

        Returns:
            str: string represeting the metadata context
        """
        
        if len(self.metadata_context) == 0:
            return 'no-metadata'
        
        s = '_'.join([y for _, y in self.metadata_context.items()])
        s = s.replace(" ", "_")
        return s
        
    def __str__(self) -> str:
        """Get called when the object is printed

        Returns:
            str: string to be printed
        """
        return 'Generation context with metadata ' + self.to_str()
    
 
 
 
def dge_name(context1 : GenerationContext, context2: GenerationContext) -> str:
    """
    DGE file name between two generation contexts.

    Args:
        context1 (GenerationContext): The first generation context.
        context2 (GenerationContext): The second generation context.

    Returns:
        str: A string representing the DGE file name, formatted as 'context1_vs_context2'.
    """
    return context1.to_str() + '_vs_' + context2.to_str()


class GenerativeModel(nn.Module):
    """GenerativeModel class."""
    
    def __init__(self, gene_list: list[str], gene_symbols: list[str], metadata_names: Union[list[str], tuple[str]], **kwargs) -> None:
        """Constructor
        
        Args:
            gene_list (list[str]): gene list
            metadata_names (Union[list[str], tuple[str]]): metadata names
        """
        
        super().__init__(**kwargs)
        self.gene_list = gene_list
        self.gene_symbols = gene_symbols
        self.metadata_names = metadata_names
        self.device = None
    
    def set_device(self, device: device) -> None:
        """moves model to device and stores the device the model is on

        Args:
            device (torch.device): device
        """
        self.device = device
        self.to(device)
    
    def get_device_to_run_on(self, run_on: Optional[Union[device, Tensor]] = None) -> Union[device, Tensor]:
        """Returns device to run generation on

        Args:
            run_on (Optional[Union[torch.device, torch.Tensor]]): a device or a tensor on a device

        Raises:
            ValueError: No device set or passed as argument

        Returns:
            Union[torch.device, torch.Tensor]: device
        """
        
        if run_on is None and self.device is None:
            raise ValueError('device is not set to push the context for generation to')
        return run_on if run_on is not None else self.device
    
    
    def generate(self, context: GenerationContext, num_samples: int, batch_size: int, **kwargs) -> Union[np.ndarray, ad.AnnData]:
        """Generate samples from the model."""
        raise NotImplementedError('You should implement this method in a subclass of GenerativeModel')
    
    def raw_counts_to_model_counts(self, raw_counts: Union[np.ndarray, ad.AnnData, csc_matrix], **kwargs) -> np.ndarray:
        """Convert raw counts to model counts"""
        raise NotImplementedError('You should implement this method in a subclass of GenerativeModel')
    
    def batch_to_counts(self, x: Tensor, **kwargs) -> np.ndarray:
        """Convert A batch to counts"""
        raise NotImplementedError('You should implement this method in a subclass of GenerativeModel')
    
    def generated_counts_to_anndata(self, 
                             generated: np.ndarray, 
                             metadata: Union[GenerationContext, dict[str, str]], 
                             name: Union[str, int] = 1,
                             metadata_as_in_data: bool = True
                             ) -> ad.AnnData:
        """Convert generated data to anndata by adding metadata
        
        Args:
            generated (np.ndarray): generated counts
            metadata (Union[GenerationContext, dict[str, str]]): metadata
            name (Union[str, int], optional): name. Defaults to 1.
            metadata_as_in_data (bool, optional): keep only the metadata that was included in the training. Defaults to True.
            
        Returns:
            ad.AnnData: anndata
        """
        assert isinstance(metadata, (GenerationContext, dict)), 'metadata must be GenerationContext or dict'
        metadata = metadata.metadata_context if isinstance(metadata, GenerationContext) else metadata
        adata = ad.AnnData(generated)
        adata.var_names = self.gene_list
        adata.obs_names = [f'generated_{name}_{i}' for i in range(len(adata))]
        adata.obs['real_or_fake'] = pd.Categorical(['fake'] * len(adata))
        if metadata_as_in_data:
            for metadata_name in self.metadata_names:
                adata.obs[metadata_name] = pd.Categorical([metadata.get(metadata_name, 'unknown')] * len(adata))
        else:
            for metadata_name, metadata_value in metadata.items():
                adata.obs[metadata_name] = pd.Categorical([metadata_value] * len(adata))
        return adata
    
        