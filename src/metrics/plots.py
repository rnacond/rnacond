from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import plotly.express as px
import pandas as pd
import anndata as ad
from typing import Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from src.modules.base_generative_model import GenerationContext
from src.metrics.generation_metrics import scdesign_stats


def plot_dim_reduce(adata: ad.AnnData, 
                    method: Literal['tsne', 'umap', 'pca'], 
                    color: Optional[str] = None, 
                    symbol: Optional[str] = None, 
                    title: str = 'dim_red',
                    save: bool = False,
                    show: bool = True,
                    color_discrete_map: Optional[dict] = None,
                    symbol_map: Optional[dict] = None
                    ) -> None:
    """Computes and plots a dimensionality reduction of the data

    Args:
        adata (ad.AnnData): data
        method (Literal['tsne', 'umap', 'pca']): method to use
        color (Optional[str], optional): column name in adata to use to color data points. Defaults to None.
        symbol (Optional[str], optional): column name in adata to use to symbolize data points. Defaults to None.
        title (str, optional): to add to plot title  . Defaults to ''.
        color_discrete_map (Optional[dict], optional): mapping from color category values to colors. Defaults to None.
        symbol_map (Optional[dict], optional): mapping from symbol category values to plotly marker symbols. Defaults to None.

    Raises:
        ValueError: _description_
    """
    assert color is None or color in adata.obs.columns, f"Invalid color column {color}"
    assert symbol is None or symbol in adata.obs.columns, f"Invalid symbol column {symbol}"
    assert method in ['tsne', 'umap', 'pca'], f"Invalid method {method}, only 'tsne', 'umap' and 'pca' are allowed"
    assert save or show, "You must choose to save or show the plot"
    X = adata.X
    if method == 'tsne':
        proj = TSNE(n_components = 2, random_state = 42, init = 'random')
    elif method == 'umap':
        proj = UMAP(n_components = 2, init = 'random', random_state = 0)
    elif method == 'pca':
        proj = PCA(n_components = 2)
    X_proj = proj.fit_transform(X)
    title = method + ' ' + title
    plot_df = pd.DataFrame({'component_1': X_proj[:, 0], 'component_2': X_proj[:, 1]})
    if color is not None:
        plot_df[color] = adata.obs[color].astype(str).to_numpy()
    if symbol is not None:
        plot_df[symbol] = adata.obs[symbol].astype(str).to_numpy()
    fig = px.scatter(plot_df, x = 'component_1', y = 'component_2', color = color, symbol = symbol,
                     labels = {'component_1': 'First component', 'component_2': 'Second component',
                               'color': color, 'symbol': symbol},
                     color_discrete_map = color_discrete_map,
                     symbol_map = symbol_map)
    # Plotly can split traces by color-symbol combinations; enforce deterministic styling per trace.
    if color_discrete_map is not None:
        for trace in fig.data:
            trace_name = str(trace.name).lower()
            for key, value in color_discrete_map.items():
                if str(key).lower() in trace_name:
                    trace.marker.color = value
                    break
    if symbol_map is not None:
        for trace in fig.data:
            trace_name = str(trace.name).lower()
            for key, value in symbol_map.items():
                if str(key).lower() in trace_name:
                    trace.marker.symbol = value
                    break
    fig.update_layout(title = title, xaxis_title = "First component", yaxis_title = "Second component") 
    if show:
        fig.show()  
    if save:
        fig.write_image(title.replace(" ", "_") + ".png") 
        
        
        
def plot_rgb_correlations(correlations: np.ndarray,
                          title: str = 'corr',
                          save: bool = False,
                          show: bool = False,
                          ) -> np.ndarray:
    """Turns a (n_genes x n_genes) correlations array (greyscale) into a (n_genes x n_genes x 3) array to be 
    plotted in RGB.

    Args:
        correlations (np.ndarray): (n_genes x n_genes) correlations array

    Returns:
        np.ndarray: (n_genes x n_genes x 3) correlations array to be plotted in RGB
    """
    
    red = np.zeros_like(correlations)
    green = np.zeros_like(correlations)
    blue = np.zeros_like(correlations)

    positive = correlations > 0
    negative = correlations < 0

    red = np.where(positive, correlations, 0)
    blue = np.where(negative, -correlations, 0)

    image = np.zeros((correlations.shape[0], correlations.shape[1], 3))
    image[:,:,0] = red
    image[:,:,1] = green
    image[:,:,2] = blue
    
    plt.figure()
    plt.imshow(image)
    if show:
        plt.show()
    elif save:
        plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()
    
    return image




def scatter_plot(x: np.ndarray,
                 y: np.ndarray, 
                 x_name: str = 'x', 
                 y_name: str = 'y', 
                 fig_name: str = 'scatter.png', 
                 mini: Optional[float] = None, 
                 maxi: Optional[float] = None) -> None:
    """Plots a scatter plot of x and y
    Args:
        x (numpy.ndarray): x values
        y (numpy.ndarray): y values
        x_name (str) : x axis label
        y_name (str) : y axis label
        fig_name (str) : image name
        mini (Optional[float]) : minimum value for axis
        maxi (Optional[float]) : maximum value for axis
    Returns:
        None
    """
    maxi = max(x.max(), y.max()) if maxi is None else maxi
    mini = min(x.min(), y.min()) if mini is None else mini
    fig = plt.figure()
    plt.scatter(x, y, s = 3)
    plt.plot([mini, maxi], [mini, maxi], 'k-', color = 'r')
    plt.xlim(1.1 * mini, 1.1 * maxi)
    plt.ylim(1.1 * mini, 1.1 * maxi)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(fig_name, bbox_inches = 'tight')
    plt.close(fig)

def plot_means_and_vars(real_samples: np.ndarray, pred_samples: np.ndarray, fig_name: str = '') -> None:
    """Plots the means and variances of real and generated samples
    Args:
        real_samples (numpy.ndarray): real samples
        pred_samples (numpy.ndarray): generated samples
        fig_name (str) : image name prefix
    Returns:  
        None
    """
    real_means, pred_means = np.mean(real_samples, axis = 0), np.mean(pred_samples, axis = 0)
    scatter_plot(real_means, pred_means, 'real mean counts', 'pred mean counts', fig_name + 'means.png')
    real_vars, pred_vars = np.var(real_samples, axis = 0), np.var(pred_samples, axis = 0)
    scatter_plot(real_vars, pred_vars, 'real count variance', 'pred count variance', fig_name + 'vars.png')

def roc(x: np.ndarray, y: np.ndarray, fig_name: str = 'roc.png') -> None:
    """Plots the ROC curve and compute AUROC
    Args:
        x (numpy.ndarray): x values
        y (numpy.ndarray): y values
        fig_name (str) : image name
    Returns:
        None
    """
    auc = np.trapz(x, y)
    plt.figure()
    plt.plot(x, y)
    plt.title('AUC: ' + str(auc))
    plt.savefig(fig_name, bbox_inches = 'tight')
    plt.close()
    
    
def plot_logfoldchanges(log2foldchanges1: np.ndarray, log2foldchanges2: np.ndarray) -> None:
    """Computes spearman and pearson correlations
    Args:
        log2foldchanges1 (numpy.ndarray): ground truth log fold changes
        log2foldchanges2 (numpy.ndarray): predicted log fold changes
    Returns:
        None
    """
    plt.scatter(log2foldchanges1, log2foldchanges2)
    plt.plot([-1.5, 3.3], [-1.5, 3.3], 'k-', color = 'r')
    plt.xlabel('Ground truth log fold changes')
    plt.ylabel('Predicted log fold changes')
    plt.savefig('logfold-change.png', bbox_inches = 'tight')
    
    
def qq_plot(array1: np.ndarray, 
            array2: np.ndarray, 
            name1: str = 'x', 
            name2: str = 'y', 
            name: str = 'qq',
            save: bool = True,
            show: bool = False
            ) -> None:
    """qq plot

    Args:
        array1 (np.ndarray): array1
        array2 (np.ndarray): array2
        name1 (str): axis 1 name
        name2 (str): axis 2 name
        name (str): figure name
        save (bool): save plot or not
        show (bool): show plot or not
     Returns:
        None   
        
    """
    
    means1 = scdesign_stats(array1)['mean_genes']
    means2 = scdesign_stats(array2)['mean_genes']
    m = max([np.max(means1), np.max(means2)])
    fig, ax = plt.subplots()
    ax.scatter(np.sort(means1), np.sort(means2))
    ax.plot([0, m + 2], [0, m + 2], color = 'black', linewidth = 2)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_box_aspect(3 / 2)
    if show:
        plt.show()
    if save:
        plt.savefig(name.replace(" ", "_") + ".png", bbox_inches = 'tight')
    plt.close()
    
    
def pairplot(counts1: np.ndarray, 
             counts2: np.ndarray, 
             context1: dict[str, str],
             context2: dict[str, str],
             name: str = 'pairplot',
             save: bool = True,
             show: bool = False
             ) -> None:
    """Parplots of the scDesign stats between two conditions

    Args:
        counts1 (np.ndarray): counts
        counts2 (np.ndarray): counts
        context1 (dict[str, str]): context for counts1
        context2 (dict[str, str]): context for counts2
        name (str): figure name
        save (bool) save figure or not
        show (bool): show figure or not
    """
    
    stats1 = scdesign_stats(counts1)
    stats2 = scdesign_stats(counts2)
    stats1['condition'] = GenerationContext(metadata_context = context1).to_str()
    stats2['condition'] = GenerationContext(metadata_context = context2).to_str()
    df_real = pd.concat([pd.DataFrame(stats1), pd.DataFrame(stats2)]).reset_index(drop = True)
    q = seaborn.pairplot(df_real, hue = 'condition')
    if show:
        print(q)
    if save:
        q.savefig(name.replace(" ", "_") + '.png')
    
    
def gene_mean_plot(array1: np.ndarray, 
                   array2: np.ndarray, 
                   name1: str = 'x', 
                   name2: str = 'y', 
                   name: str = 'qq',
                   save: bool = True,
                   show: bool = False,
                   annotate: list[int] = [],
                   ) -> None:
    """qq plot

    Args:
        array1 (np.ndarray): array1
        array2 (np.ndarray): array2
        name1 (str): axis 1 name
        name2 (str): axis 2 name
        name (str): figure name
        save (bool): save plot or not
        show (bool): show plot or not
     Returns:
        None   
        
    """
    
    means1 = scdesign_stats(array1)['mean_genes']
    means2 = scdesign_stats(array2)['mean_genes']
    m = max([np.max(means1), np.max(means2)])
    plt.figure()
    plt.scatter(means1, means2)
    for i in range(len(means1)):
        if i in annotate:
            plt.annotate(str(i+1), (means1[i], means2[i]))
    plt.plot([0, m + 2], [0, m + 2], color = 'black', linewidth = 2)
    plt.xlabel(name1)
    plt.ylabel(name2)
    if show:
        plt.show()
    if save:
        plt.savefig(name.replace(" ", "_") + ".png", bbox_inches = 'tight')
    plt.close()