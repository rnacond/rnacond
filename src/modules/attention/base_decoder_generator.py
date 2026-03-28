import numpy as np
import torch
from typing import Optional, Union
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer
from transformers import DistilBertTokenizer
import torch.nn.functional as F
from src.utils.utils import padded_stack
import math
from src.dataops.data_utils import sequences_to_counts, sample_counts, sequences_to_anndata
import numpy as np
from src.modules.base_generative_model import GenerationContext, GenerativeModel
from torch import Tensor, device
import anndata as ad

class DecoderGenerator(GenerativeModel):
    """DecoderGenerator. Contains generate method to generate causally"""
    
    def __init__(self, 
                 gene_list: list[str],
                 gene_symbols: list[str],
                 gene_tokenizer: GeneTokenizer, 
                 block_size: int = 1500,
                 metadata_names: Union[list[str], tuple[str]] = [],
                 sample: bool = False,
                 num_total_counts: Optional[int] = None,
                 text_tokenizer: Optional[DistilBertTokenizer] = None,
                 **kwargs
                 ) -> None:
        """Constructor
        
        Args:
            gene_list (list[str]): gene list
            gene_symbols (list[str]): gene symbols
            gene_tokenizer (Tokenizer): gene tokenizer
            block_size (int): block size
            metadata_names (Union[list[str], tuple[str]]): metadata names
            sample (bool): sample or not (in the training data)
            num_total_counts (Optional[int]): number of total counts sampled in sample is True (in the training data)
            text_tokenizer (Optional[DistilBertTokenizer]): text tokenizer
        """
        
        super().__init__(gene_list, gene_symbols, metadata_names, **kwargs)
        self.gene_tokenizer = gene_tokenizer
        self.text_tokenizer = text_tokenizer
        self.tokens = self.gene_tokenizer.get_tokens() 
        self.block_size = block_size
        self.sample = sample
        self.num_total_counts = num_total_counts
        self.pad_idx = self.gene_tokenizer.special_tokens_ids['pad_token']
        self.end_idx = self.gene_tokenizer.special_tokens_ids['end_token']
        self.start_token = self.gene_tokenizer.special_tokens['start_token']
        self.unknown_token = self.gene_tokenizer.special_tokens['unknown_token']
        self.gene_start_token = self.gene_tokenizer.special_tokens['gene_start_token']
        self.gene_order = self.gene_tokenizer.gene_order
        
    
    def encode_context(self, context: GenerationContext, batch_size: int = 8, run_on: Optional[Union[device, Tensor]] = None) -> dict[str, Tensor]:
        """Encodes the context for generation

        Args:
            context (GenerationContext): contains 'gene_context' and maybe 'text_context'
            batch_size (int): batch size, defaults to 8
            run_on (Optional[Union[torch.device, torch.Tensor]]): device to run on, if None will use self.device if set

        Returns:
            Dict[str, torch.Tensor]: contains the input to get_scores, must contain 'gene_ids' 
        """
        
        raise NotImplementedError("You should implement this method in a subclass of DecoderGenerator")    

    def get_scores(self, gene_ids: Tensor, *args, **kwargs) -> Tensor:
        """Gets scores for generation
        
        Args:
            gene_ids (torch.Tensor): gene_ids
            *args: args
            **kwargs: kwargs
            
        Returns:
            torch.Tensor: scores
        """
        
        raise NotImplementedError("You should implement this method in a subclass of DecoderGenerator")
    
    
    def _generate_batch(self, encoded_context: dict[str, Tensor], batch_size: int = 8, max_new_tokens: int = 2000) -> Tensor:
        """Given a context, generates B samples of size T where B = batch_size and T = context_length + max_new_tokens

        Args:
            encoded_context (dict[str, Tensor]): context, input to get_scores, must contain 'gene_ids'
            batch_size (int): batch size, defaults to 8
            max_new_tokens (int): Number of tokens to generate
            
        Returns:
            torch.Tensor: generation output, a B x T tensor 
        """
        
        assert 'gene_ids' in encoded_context, 'gene_ids missing from encoded context'
        i, idx, input = 0, encoded_context['gene_ids'], encoded_context.copy()
        ended: list[Tensor] = []
        
        while True:
            i += 1    
            # crop idx to the last block_size tokens
            input['gene_ids'] = idx[:, - self.block_size:]
            # get the predictions
            with torch.no_grad():
                scores = self.get_scores(**input)
            # focus only on the last time step
            scores = scores[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(scores, dim = -1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
            # stop generating for batch elements with end token
            just_ended = (idx_next == self.end_idx).T[0]
            if just_ended.sum() > 0:
                ended.append(idx[just_ended, :])
                idx = idx[~just_ended, :]
            for key in encoded_context:
                input[key] = input[key][~just_ended, :]
            if (sum([x.size(0) for x in ended]) == batch_size and idx.size(0) == 0) or i == max_new_tokens:
                break
        
        idx = padded_stack(ended + [idx], value = self.pad_idx)[0]
        return idx
    
    def generate(self, 
                 context: GenerationContext, 
                 num_samples: int = 8, 
                 batch_size: int = 8, 
                 max_new_tokens: int = 2000, 
                 verbose: bool = True, 
                 run_on: Optional[Union[torch.device, torch.Tensor]] = None,
                 ) -> ad.AnnData:
        
        """Given a context, generates N samples of size T where N = num_samples and T = context_length + max_new_tokens

        Args:
            context (GenerationContext): Context to generate from, contains 'gene_context' and 'text_context'
            num_samples (int): how many samples to generate, defaults to 8
            batch_size (int): batch size, defaults to 8
            max_new_tokens (int): Number of tokens to generate
            verbose (bool): print a generated sequence or not
            run_on (Union[torch.device, torch.Tensor]): device to run on, if None will use self.device if set

        Returns:
            ad.AnnData: generation output 
        """
        
        run_on = self.get_device_to_run_on(run_on)
        encoded_context = self.encode_context(context, batch_size, run_on)
        max_new_tokens = (self.block_size + 1 - encoded_context['gene_ids'].size(-1)) if self.sample else max_new_tokens

        if 'gene_ids' not in encoded_context:
            raise ValueError('gene_ids missing from encoded context')
        
        num_batches = int(math.ceil(num_samples / batch_size))
        generated_idx = []
        
        for _ in range(num_batches):
            generated_idx.append(self._generate_batch(encoded_context.copy(), batch_size, max_new_tokens))
            
        generated_idx = padded_stack(generated_idx, value = self.pad_idx)[0][:num_samples, :]
        generated_counts = self.batch_to_anndata(generated_idx, verbose)
        return generated_counts
    
    def batch_to_counts(self, x: Tensor, verbose: bool = False) -> np.ndarray:
        """Converts an output tensor (a sequence of token ids) to counts

        Args:
            x (torch.Tensor): tensor of token ids
            verbose (bool): print a generated sequence or not

        Returns:
            np.ndarray: counts
        """

        x_seq = self.gene_tokenizer.detokenize_batch(x) 
        x_counts = sequences_to_counts(x_seq, self.gene_list)
        if verbose:
            print('Generation example\n', x_counts, '\nGenerated counts sum per gene', x_counts.sum(0), '\nGenerated counts sums', x_counts.sum(1), 'shape', x_counts.shape, flush = True)
        return x_counts
    
    def batch_to_anndata(self, x: Tensor, verbose: bool = False) -> ad.AnnData:
        """Converts an output tensor (a sequence of token ids) to counts

        Args:
            x (torch.Tensor): tensor of token ids
            verbose (bool): print a generated sequence or not

        Returns:
            ad.AnnData: counts and metadata
        """

        x_seq = self.gene_tokenizer.detokenize_batch(x) 
        x_adata = sequences_to_anndata(x_seq, self.gene_list, self.gene_tokenizer.metadata_tokens_per_column)
        if verbose:
            x_counts = x_adata.X
            print('Generation example\n', x_counts, '\nGenerated counts sum per gene', x_counts.sum(0), '\nGenerated counts sums', x_counts.sum(1), 'Shape', x_counts.shape, 'Metadata', x_adata.obs, flush = True)
        return x_adata

    def raw_counts_to_model_counts(self, raw_counts: ad.AnnData) -> np.ndarray:
        """Converts raw counts to model counts by imitating the data sampling process to evaluate generation

        Args:
            raw_counts (ad.AnnData): raw counts

        Returns:
            np.ndarray: model counts
        """
        
        raw_counts = raw_counts[:, self.gene_list]
        return sample_counts(raw_counts, self.num_total_counts) if self.sample else raw_counts.X.toarray()
    
    

        