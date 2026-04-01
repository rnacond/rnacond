from src.metrics.generation_metrics import test_classification_on_fake_data, distances

from typing import Union
import numpy as np
import anndata as ad
import scanpy as sc

from src.metrics.plots import plot_rgb_correlations
from src.metrics.generation_metrics import pearsonr

from src.utils.utils import to_array

from src.dataops.data_utils import extract, counts_to_anndata, concat_real_and_fake_anndata

from src.metrics.plots import plot_dim_reduce, qq_plot, pairplot

from src.modules.base_generative_model import GenerationContext


def evaluation_(real_counts: np.ndarray, 
                fake_counts: np.ndarray, 
                prompt_name: str, 
                model_name: str, 
                n_genes: int,
                save: bool,
                show: bool) -> None:
    
    """ Runs metrics and plots between real and fake data from same prompt

    Args:
        real_counts (np.ndarray): real counts
        fake_counts (np.ndarray): fake counts
        prompt_name (str): name of the prompt for which the data was generated
        model_name (str, optional): Model name that generated the fake counts. Defaults to 'Model'.
        n_genes (int): number of genes for correlation plots
        save (bool, optional): save plots or not
        show (bool, optional): show plots or not in notebook
    """
    
    print('\n ----- Distances between real and fake data with prompt', prompt_name)
    print(distances(real_counts, fake_counts))
    title = model_name + '_corrleations_real_' + prompt_name.replace(' ', '_') 
    print(title)
    plot_rgb_correlations(np.nan_to_num(pearsonr(real_counts[:, :n_genes])), title, save, show)
    title = model_name + '_corrleations_fake_' + prompt_name.replace(' ', '_') 
    print(title)
    plot_rgb_correlations(np.nan_to_num(pearsonr(fake_counts[:, :n_genes])), title, save, show)
    title = model_name + '_qq_plot_' + prompt_name.replace(' ', '_') + '.png'
    qq_plot(real_counts, fake_counts, 'Real gene means', model_name + ' generated gene means', title, save, show)




def evaluation(fake_counts: list[Union[np.ndarray, ad.AnnData]], 
               prompts: list[dict[str, str]],
               real_adata: ad.AnnData,
               column: str,
               n_genes: int = 30,
               save: bool = True,
               show: bool = False,
               model_name: str = 'Model',
               ) -> None:
    
    """ Runs metrics and plots between real and fake data for multiple prompts.

    Args:
        fake_counts (list[Union[np.ndarray, ad.AnnData]]): list of generated data
        prompts (list[dict[str, str]]): list of prompts for generated data in same order
        real_adata (ad.AnnData): must contain genes in same order as fake counts and cells corresponding to the prompts
        column (str): column in adata.obs and prompts for umap (e.g cell type or disease)
        n_genes (int, optional): number of genes for correlation plots
        save (bool, optional): save plots or not
        show (bool, optional): show plots or not in notebook
        model_name (str, optional): Model name that generated the fake counts. Defaults to 'Model'.
    """
    
    assert len(prompts) == len(fake_counts)
    assert sum([column in prompt for prompt in prompts]) == len(prompts)
    assert save or show
        
    contexts = [GenerationContext(metadata_context = prompt) for prompt in prompts]
    contexts_names = [context.to_str() for context in contexts]
    
    fake_counts = [to_array(f) for f in fake_counts]
    fake_counts_all = np.vstack(fake_counts)    
    fake_adatas = [counts_to_anndata(f, real_adata.var_names, prompts[i]) for i, f in enumerate(fake_counts)]
    fake_adata = ad.concat(fake_adatas, keys = ['p' + str(i) for i in range(len(prompts))], index_unique = '_')
    
    real_adatas_all = [extract(real_adata, p).copy() for p in prompts]
    real_counts = [a.X.toarray()[:fake_counts[i].shape[0], :] for i, a in enumerate(real_adatas_all)]
    real_counts_all = np.vstack(real_counts)
    real_adata_all = ad.concat(real_adatas_all, keys = ['p' + str(i) for i in range(len(prompts))], index_unique = '_')
    
    print('\n ----- Evaluating model', model_name)
    
    sc.pl.heatmap(fake_adata, fake_adata.var_names, groupby = column, swap_axes = True, vmax = 3.0)
    sc.pl.heatmap(real_adata_all, real_adata_all.var_names, groupby = column, swap_axes = True, vmax = 3.0)
    evaluation_(real_counts_all, fake_counts_all, 'all', model_name, n_genes, save, show)
    
        
    
    for i, name in enumerate(contexts_names):
        evaluation_(real_counts[i], fake_counts[i], name, model_name, n_genes, save, show)
        
        
    real_and_fake_adata = concat_real_and_fake_anndata(real_adata_all, fake_adata)
    real_and_fake_adata.obs['real_or_fake'] = real_and_fake_adata.obs['real_or_fake'].astype(str).str.lower()
    disease_values = [str(v) for v in real_and_fake_adata.obs[column].unique()]
    symbol_map = {v: ('circle' if 'crohn' in v.lower() else 'diamond') for v in disease_values}
    plot_dim_reduce(real_and_fake_adata, 'umap', 'real_or_fake', column, 'umap column', save, show,
                    color_discrete_map={'real': 'blue', 'fake': 'red'},
                    symbol_map=symbol_map)
    
    if len(prompts) > 1:
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                name = contexts_names[i] + '_' + contexts_names[j]
                print('\n ----- Training on fake data for classifying', contexts_names[i], 'vs', contexts_names[j])
                test_classification_on_fake_data(real_counts[i], real_counts[j], fake_counts[i], fake_counts[j])
                pairplot(real_counts[i], real_counts[j], prompts[i], prompts[j], 'pairplot_real_' + name, save, show)
                pairplot(fake_counts[i], fake_counts[j], prompts[i], prompts[j], 'pairplot_fake_' + name, save, show)

                

def evaluation_from_paths(fake_counts: list[str], 
                          prompts: list[dict[str, str]],
                          real_adata: str,
                          column: str,
                          n_genes: int = 30,
                          save: bool = True,
                          show: bool = False,
                          model_name: str = 'Model') -> None:
    
    """ Runs metrics and plots between real and fake data for multiple prompts.

    Args:
        fake_counts (list[str]): list of paths to generated data 
        prompts (list[dict[str, str]]): list of prompts for generated data in same order
        real_adata (str): path to real data in h5ad format
        column (str): column in adata.obs and prompts for umap (e.g cell type or disease)
        n_genes (int, optional): number of genes for correlation plots
        save (bool, optional): save plots or not
        show (bool, optional): show plots or not in notebook
        model_name (str, optional): Model name that generated the fake counts. Defaults to 'Model'.
    """
    
    fake_counts = [ad.read_h5ad(f) if f.endswith('.h5ad') else np.load(f) for f in fake_counts]
    real_adata = ad.read_h5ad(real_adata)
    evaluation(fake_counts, prompts, real_adata, column, n_genes, save, show, model_name)
    