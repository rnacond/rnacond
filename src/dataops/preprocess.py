from typing import Union, Optional
import numpy as np
import anndata as ad
import anndata as ad
import scanpy as sc
from pydantic import BaseModel, ConfigDict
from src.utils.utils import to_array, to_config
from src.dataops.data_utils import extract
from scipy import sparse

    
class PreprocessConfig(BaseModel):
    model_name: Optional[str] = None
    gene_name_col: Optional[str] = None # Use a column in the adata as gene names
    variable_genes: Optional[int] = None # Select this many most highly variable genes
    max_total_counts: Optional[int] = None # Remove cells with more than this many total counts
    min_genes: Optional[int] = None # remove cell with less than this many genes
    min_cells: Optional[int] = None # remove genes with less than this many genes
    normalize: bool = False
    target_sum: Optional[int] = None # Normalize to sum to this (median if None)
    round: bool = False
    log: bool = False # Log values
    log_base: int = 2
    only_keep_prompts: bool = False
    prompts: list[dict[str, str]] = [{}]
    model_config = ConfigDict(protected_namespaces = ())
    
    

def only_keep_prompts(adata: ad.AnnData, prompts: list[dict[str, str]] = [{}]) -> ad.AnnData:
    """extracts and keeps only the cell with the metadata used as prompts for evaluation

    Args:
        adata (ad.AnnData): data
        prompts (list[dict[str, str]], optional): metadata prompts. Defaults to [{}].

    Returns:
        ad.AnnData: extracted data
    """
    adatas = [extract(adata, prompt) for prompt in prompts if len(prompt) > 0]
    return ad.concat(adatas) if len(adatas) > 0 else adata

def change_gene_index(adata: ad.AnnData, gene_name_col: str) -> ad.AnnData:
    """Change gene index
    
    Args:
        adata (ad.AnnData): data
        gene_name_col (str, optional): gene name column.
        
    Returns:
        ad.AnnData: preprocessed data
    """
    
    if gene_name_col not in adata.var.columns:
        print('[!] Provided gene_name_col', gene_name_col, 'not in adata.var.columns. Using index.')
        return adata
    else:
        adata = adata[:, ~adata.var[gene_name_col].duplicated()]
        adata.var_names = adata.var[gene_name_col].astype(str).values
        adata.var_names.name = 'index'
        print('[X] Changed gene index to', gene_name_col)
        return adata

def check_metadata_names(adata: ad.AnnData, metadata_names: list[str]) -> ad.AnnData:
    """Checks that the metadata names are in the adata and selects them
    
    Args:
        adata (ad.AnnData): data
        metadata_names (list[str]): metadata names
        
    Raise:
        ValueError: if metadata names are not in adata
        
    Returns:
        ad.AnnData: adata with only the obs columns in metadata names
    """
    
    for metadata_name in metadata_names:
        if metadata_name not in adata.obs.keys():
            raise ValueError('The metadata name', metadata_name, 'is not in the adata object. The available metadata names are:', adata.obs.columns.tolist())
    print('\n[X] Checked that all metadata names are in the adata object.')
    # adata.obs = adata.obs.fillna('nan')
    for z in adata.obs.columns:
        adata.obs[z] = adata.obs[z].astype('category')
    return adata
    
def select_highly_variable_genes(adata : ad.AnnData, n_genes: int) -> ad.AnnData:
    """selects highly variable genes

    Args:
        adata (ad.AnnData): raw data (counts)
        n_genes (int): number of genes to keep, if None do nothing

    Returns:
        ad.AnnData: adata with the n_genes most highly variable genes
    """
    
    print('[X] Selecting the', n_genes, 'most highly variable genes.')
    adata_log = sc.pp.log1p(adata, copy = True)
    highly_variable_genes = sc.pp.highly_variable_genes(adata_log, n_top_genes = n_genes, inplace = False)
    return adata[:, highly_variable_genes['highly_variable']].copy()


def total_counts_cutoff(adata: ad.AnnData, max_total_counts: int) -> ad.AnnData:
    """Keep cells below a certain total count

    Args:
        adata (ad.AnnData): data
        max_total_counts (int): max total counts

    Returns:
        ad.AnnData: data with only cells with at most max_total_counts counts
    """
    
    if 'log1p_total_counts' not in adata.obs.columns:
        raise ValueError('log1p_total_counts not in data')
    adata.obs['total_counts'] = list((map(int, np.round(np.exp(adata.obs['log1p_total_counts'].to_numpy()) - 1).tolist())))
    adata = adata[adata.obs.total_counts <= max_total_counts]
    return adata
    



def preprocess_adata(adata: ad.AnnData, 
                     metadata_names: list[str], 
                     preprocess_config: Union[dict[str, Union[None, int, str]], PreprocessConfig]
                     ) -> ad.AnnData:
    
    """Preprocessing

    Args:
        adata (ad.AnnData): data
        metadata_names (list[str]): metadata names to check if they are in adata
        preprocess_config (Union[dict[str, Union[None, int, str]], PreprocessConfig]): preprocessing config

    Returns:
        ad.AnnData: data after preprocessing
    """
    
    adata.X = np.nan_to_num(adata.X, nan = 0)
    adata.X = sparse.csc_matrix(adata.X)
    preprocess_config = to_config(preprocess_config, PreprocessConfig)
    adata = check_metadata_names(adata, metadata_names)
    if preprocess_config.gene_name_col is not None:
        adata = change_gene_index(adata, preprocess_config.gene_name_col) 
    if preprocess_config.min_genes is not None:
        sc.pp.filter_cells(adata, min_genes = preprocess_config.min_genes)
    if preprocess_config.min_cells is not None:
        sc.pp.filter_genes(adata, min_cells = preprocess_config.min_cells)
    if preprocess_config.variable_genes is not None:
        adata = select_highly_variable_genes(adata, preprocess_config.variable_genes) 
    if preprocess_config.max_total_counts is not None:
        adata = total_counts_cutoff(adata, preprocess_config.max_total_counts) 
    if preprocess_config.normalize:
        sc.pp.normalize_total(adata, target_sum = preprocess_config.target_sum) 
        adata.X = to_array(adata.X)
        if preprocess_config.round:
            adata.X = np.where((adata.X > 0) & (adata.X < 0.5), 1, adata.X)
            adata.X = adata.X.round() if not isinstance(adata.X, np.ndarray) else np.round(adata.X)
    if preprocess_config.log:
        sc.pp.log1p(adata, base = preprocess_config.log_base)
    if preprocess_config.only_keep_prompts:
        adata = only_keep_prompts(adata, preprocess_config.prompts)
    return adata