import pandas as pd
from typing import Union
from torch import Tensor
import numpy as np
import anndata as ad
import random
from src.utils.utils import padded_stack
from scipy import sparse
import torch
import anndata as ad
from scipy.sparse import csc_matrix
from typing import Optional

def train_val_split(adata: ad.AnnData, percent_train: float = 0.9) -> tuple[ad.AnnData, ad.AnnData]:
    """Splits adata into train and validation sets

    Args:
        adata (ad.AnnData): data
        percent_train (float, optional): percentage of data to use for training. Defaults to 0.9.

    Returns:
        Tuple[ad.AnnData, ad.AnnData]: train and validation sets
    """
    train_size = int(percent_train * len(adata))
    train_adata = adata[:train_size]
    valid_adata = adata[train_size:]
    return train_adata, valid_adata

def sequences_to_anndata(sequences: list[list[str]], 
                         gene_order: Union[dict[str, int], list[str]],
                         metadata_tokens_per_column: dict[str, list[str]]) -> ad.AnnData:
    """sequences of genes to gene counts

    Args:
        sequences (list[list[str]]): batch of sequences
        gene_order(Union[dict[str, int], list[str]]): gene to index dict or gene list in the same order
        metadata_tokens_per_column (dict[str, list[str]]): known metadata tokens 

    Returns:
        ad.AnnData: counts of size len(sequences) x len(gene_order), and metadata
    """
    counts = []
    metadata = {metadata_column : [] for metadata_column in metadata_tokens_per_column.keys()}
    for sequence in sequences:
        counts_, metadata_ = sequence_to_anndata(sequence, gene_order, metadata_tokens_per_column)
        counts.append(counts_)
        for key, value in metadata_.items():
            metadata[key].append(value)
            
    adata = ad.AnnData(np.vstack(np.array(counts)))
    adata.obs = pd.DataFrame.from_dict(metadata , orient = 'index').transpose()
    return adata

def sequence_to_anndata(sequence: list[str], 
                        gene_order: Union[dict[str, int], list[str]],
                        metadata_tokens_per_column: Optional[dict[str, list[str]]] = None
                        ) -> tuple[list[int], Union[None, dict[str, list[str]]]]:
    """counts genes in a sequence of genes

    Args:
        sequence (list[str]): sequence
        gene_order(Union[dict[str, int], list[str]]): gene to index dict or gene list in the same order
        metadata_tokens_per_column (Optional[dict[str, list[str]]]): known metadata tokens 
    Returns:
        list[int]: list of gene counts
        Union[None, dict[str, list[str]]]]: metadata if metadata_tokens_per_column is provided
    """
    gtoi = gene_order if isinstance(gene_order, dict) else { gene : i for i, gene in enumerate(gene_order) }
    counts = [0] * len(gtoi)
    metadata = {metadata_column : 'unknown' for metadata_column in metadata_tokens_per_column.keys()} if metadata_tokens_per_column is not None else None
    for token in sequence:
        idx = gtoi.get(token)
        if idx is not None:
            counts[idx] += 1
        if metadata_tokens_per_column is not None:
            for metadata_column, metadata_column_tokens in metadata_tokens_per_column.items():
                if token in metadata_column_tokens:
                    metadata[metadata_column] = token
    return counts, metadata

def sequence_to_counts(sequence: list[str], gene_order: Union[dict[str, int], list[str]]) -> list[int]:
    """counts genes in a sequence of genes

    Args:
        sequence (list[str]): sequence
        gene_order(Union[dict[str, int], list[str]]): gene to index dict or gene list in the same order

    Returns:
        list[int]: list of gene counts
    """
    return sequence_to_anndata(sequence, gene_order)[0]

def sequences_to_counts(sequences: list[list[str]], gene_order: Union[dict[str, int], list[str]]) -> np.ndarray:
    """sequences of genes to gene counts

    Args:
        sequences (list[list[str]]): batch of sequences
        gene_order(Union[dict[str, int], list[str]]): gene to index dict or gene list in the same order

    Returns:
        np.ndarray: counts of size (len(sequences), len(gene_list))
    """
    return np.vstack([np.array(sequence_to_counts(sequence, gene_order)) for sequence in sequences])

def sample_counts(adata: ad.AnnData, num_total_counts: int) -> np.ndarray:
    """Sample num_total_counts genes from adata

    Args:
        adata (ad.AnnData): data
        num_total_counts (int): total counts to sample

    Returns:
        np.array: sampled counts
    """
    gene_list = adata.var_names.tolist()
    counts = adata.X.toarray()
    all_sampled_counts = []
    for i in range(counts.shape[0]):
        sampled_genes = random.choices(gene_list, weights = counts[i], k = num_total_counts)
        sampled_counts = np.array(sequence_to_counts(sampled_genes, gene_list))
        all_sampled_counts.append(sampled_counts)
    return np.vstack(all_sampled_counts)

def extract(adata: ad.AnnData, conditions: Optional[dict[str, str]], n_max: Optional[int] = None) -> ad.AnnData:
    """Extract from adata based on metadata conditions

    Args:
        adata (ad.AnnData): data
        conditions (Optional[dict[str, str]]): dictionary of field name to field value conditions
        n_max (Optional[int]): max number of samples to extract. Return everything if None

    Returns:
        ad.AnnData: extracted counts
    """
    
    if conditions is None:
        adata = adata
    else:
        for i, (key, value) in enumerate(conditions.items()):
            if key not in adata.obs.keys():
                raise ValueError(f'Cannot extract, {key} not in adata obs')
            mask = adata.obs[key] == value
            if i == 0:
                total_mask = mask
            else:
                total_mask = total_mask & mask
        adata = adata if len(conditions) == 0 else adata[total_mask]
    length = len(adata)
    if length == 0:
        raise ValueError('No data after extraction')
    return adata[random.sample(range(length), n_max)] if n_max is not None and length > n_max else adata

def pad_and_collate(batch: list[dict[str, Tensor]], pad: int) -> dict[str, Tensor]:
    """Collate function for with padding, text_ids and text_attention are not padded

    Args:
        batch (list[dict[str, Tensor]]): list of samples
        pad (int): padding value

    Returns:
        dict[str, Tensor]: batch
    """
    
    keys = list(batch[0].keys())
    gene_keys = [key for key in keys if 'text_' not in key]
    
    assert 'padding_mask' not in keys, 'padding_mask already in the batch'
    assert sum([list(batch[i].keys()) == keys for i in range(len(batch))]) == len(batch), 'All samples should have the same keys'
    assert sum([sample[key].size() == sample[keys[0]].size() for sample in batch for key in gene_keys]) == len(batch) * len(gene_keys), 'All tensors in a sample should have the same size'
    
    collated_batch = dict()
    for key in keys:
        samples = [sample[key] for sample in batch]
        if key == 'mask':
            collated_batch[key], collated_batch['padding_mask'] = padded_stack(samples, value = False)
        elif key in gene_keys:
            collated_batch[key], collated_batch['padding_mask'] = padded_stack([sample[key] for sample in batch], value = pad)
        else:
            collated_batch[key] = torch.vstack(samples)
    return collated_batch


def counts_to_numpy(a: Union[np.ndarray, csc_matrix, ad.AnnData]) -> np.ndarray:
    """Converts counts to numpy array

    Args:
        a (Union[np.ndarray, csc_matrix, ad.AnnData]): counts

    Returns:
        np.ndarray: counts
    """
    assert isinstance(a, (np.ndarray, csc_matrix, ad.AnnData)), "Invalid type, must be np.ndarray, csc_matrix or AnnData"
    a = a.X if isinstance(a, ad.AnnData) else a
    a = a.toarray() if isinstance(a, csc_matrix) else a
    return a

def counts_to_anndata(counts: np.ndarray, 
                      gene_names: Optional[list[str]] = None, 
                      metadata: Optional[dict[str, str]] = None
                      ) -> ad.AnnData:
    """Puts the counts in an anndata objects with gene names and metadata

    Args:
        counts (np.ndarray): counts
        gene_names (Optional[list[str]], optional): gene names
        metadata (Optional[dict[str, str]], optional): metadata

    Returns:
        ad.AnnData: anndata
    """
    adata = ad.AnnData(counts)
    if gene_names is not None:
        adata.var_names = gene_names
    if metadata is not None:  
        for key, item in metadata.items(): 
            adata.obs[key] = pd.Categorical([item] * len(adata))  
    return adata

def concat_real_and_fake_anndata(real_adata: ad.AnnData, fake_adata: ad.AnnData) -> ad.AnnData:
    """Concatenates real and fake anndata by adding a column 'real_or_fake' to the obs if not there

    Args:
        real_adata (ad.AnnData): real data
        fake_adata (ad.AnnData): fake data

    Returns:
        ad.AnnData: concatenated anndata
    """
    assert real_adata.var_names.tolist() == fake_adata.var_names.tolist(), 'gene lists should match between real and fake data'
    
    real_adata.obs['real_or_fake'] = pd.Categorical(['real'] * len(real_adata))   
    fake_adata.obs['real_or_fake'] = pd.Categorical(['fake'] * len(fake_adata))   
    adata = ad.concat([real_adata, fake_adata], keys = ['real', 'fake'], index_unique = '_')
    assert not adata.obs.isnull().values.any(), 'There are null values in the concatenated adata'
    return adata
    
    
def count_check(X: Union[np.ndarray, sparse.csr_matrix], check_max: bool = False) -> None:
    """Check that the matrix has integer counts
    Args:
        X (Union[np.ndarray, csr_matrix]): matrix
        check_max (bool): checks that the values are large enough
    Raises:
        ValueError: Non-integer counts detected. Need to select a layer with raw counts.
        ValueError: Low value returned for sum of counts across genes. Need to check that raw counts are passed in.
     Returns:
        None
    """
    X0 = X
    if not isinstance(X0, np.ndarray):
        X0 = X0.toarray()
    if (np.rint(X0) != X0).any():
        raise ValueError("Counts layer contains non-integer values: Please select a layer with raw reads !")
    if check_max and X.sum(axis = 1).max() < 1000:
            raise ValueError("Max counts value should be at least 1000 : Please verify that layer is set to raw (integer) counts !")
        
        


       