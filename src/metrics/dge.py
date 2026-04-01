import numpy as np
import warnings 
import pandas as pd 
import matplotlib.pyplot as plt
import time
from typing import Optional, Union
from scipy import stats
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from IPython.utils import io
from src.utils.utils import extract_sublist
from src.dataops.data_utils import count_check
import gseapy as gp


np.seterr(all = 'ignore')
CONDITION_NAME = 'Condition'
DESIGN_FACTORS = [CONDITION_NAME]
CONDITION1_NAME = 'Condition 1'
CONDITION2_NAME = 'Condition 2'
GENE_SETS = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human']
warnings.filterwarnings("ignore")



def prepare_dge_data(counts1: np.ndarray, 
                     counts2: np.ndarray, 
                     condition1_name: str = CONDITION1_NAME, 
                     condition2_name: str = CONDITION2_NAME,
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares data from numpy arrays into dataframes for pydeseq2 
    Args:
        counts1 (numpy.ndarray): counts from condition 1
        counts2 (numpy.ndarray): counts from condition 2
        condition1_name (str): name of condition 1
        condition2_name (str): name of condition 2
    Returns:
        counts (pd.DataFrame): counts1 and counts2 stacked into a dataframe
        metadata (pd.DataFrame): dataframe indicating the condition
    """
    count_check(counts1)
    count_check(counts2)
    counts = pd.DataFrame(np.vstack((counts1, counts2)))
    metadata = pd.DataFrame({'Condition' : [condition1_name] * counts1.shape[0] + [condition2_name] * counts2.shape[0]})
    return counts, metadata

def dge_stats_from_dfs(counts_df: pd.DataFrame,
                       metadata_df: pd.DataFrame, 
                       design_factors: list[str] = [CONDITION_NAME], 
                       refit_cooks = False) -> pd.DataFrame:
    """Runs DESeq2 on data in dataframes and returns results df
    Args:
        counts_df (pd.DataFrame): counts
        metadata_df (pd.DataFrame): metadata
        design_factors (list[str]): design factors
        refit_cooks (bool): refit cooks or not
    Returns:
        pd.DataFrame: DGE results
    """
        
    t0 = time.time()
    print('Running DGE...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts = counts_df, metadata = metadata_df, design_factors = design_factors, refit_cooks = refit_cooks, quiet = True)
        dds.deseq2()
        stat_res = DeseqStats(dds, [CONDITION_NAME, CONDITION2_NAME, CONDITION1_NAME])
        with io.capture_output() as _ :
            stat_res.summary()
    print('Time to run DGE', time.time() - t0, 'seconds')
    return stat_res.results_df

def dge_stats_from_arrays(counts1: np.ndarray, 
                          counts2: np.ndarray, 
                          counts1_name: str = 'Condition 1', 
                          counts2_name: str = 'Condition 2', 
                          design_factors: list[str] = [CONDITION_NAME], 
                          refit_cooks: bool = False, 
                          csv_name: Optional[str] = None) -> pd.DataFrame:
    """Runs DESeq2 on data in arrays and returns results df
    Args:
        counts1 (numpy.ndarray): counts from condition 1
        counts2 (numpy.ndarray): counts from condition 2
        counts1_name (str): name of condition 1
        counts2_name (str): name of condition 2
        design_factors (list of strings): design factors
        refit_cooks (bool): refit cooks or not
        csv_name (Optional[str]): save results to csv if not None
    Returns:
        pandas.DataFrame: DGE results
    """
    counts_df, metadata_df = prepare_dge_data(counts1, counts2, counts1_name, counts2_name)
    dge_results_df = dge_stats_from_dfs(counts_df, metadata_df, design_factors, refit_cooks)
    if csv_name is not None:
        dge_results_df.to_csv(csv_name)
    return dge_results_df

def dge_genes_from_results_df(dge_results_df: pd.DataFrame, 
                              logfc_cutoff: float = 0.6, 
                              pval_cutoff: float = 0.05
                              ) -> tuple[list[int], list[int], list[int], np.ndarray, np.ndarray, list[str]]:
    """Get significant genes from DGE df results
    Args:
        dge_results_df (pd.DataFrame): DGE results
        logfc_cutoff (float): log fold change cutoff
        pval_cutoff (float): p-value cutoff
    Returns:
        up (list[int]): upregulared genes
        down (list[int]): downregulared genes
        same (list[int]): not differentially expressed genes
        log2foldchanges (numpy.ndarray): log fold changes
        pvalues (numpy.ndarray): p-values
        colors (list[str]): colors for the volcano plot
    """
    up, down, same, colors = [], [], [], []
    for i in range(dge_results_df.shape[0]):
        log2foldchange, padj = dge_results_df.iloc[i]['log2FoldChange'], dge_results_df.iloc[i]['padj']
        if log2foldchange > logfc_cutoff and padj < pval_cutoff :
            up.append(i)
            colors.append('firebrick')
        elif log2foldchange < -logfc_cutoff and padj < pval_cutoff :
            down.append(i)
            colors.append('cornflowerblue')
        else:
            same.append(i)
            colors.append('gray')
    log2foldchanges = dge_results_df['log2FoldChange'].to_numpy()
    pvalues = dge_results_df['padj'].to_numpy()
    print(len(up), 'upregulated genes', len(down), 'downregulated genes', len(same), 'insignificant genes')
    return up, down, same, log2foldchanges, pvalues, colors

def run_dge(counts1: np.ndarray,
            counts2: np.ndarray, 
            logfc_cutoff: float = 0.6, 
            pval_cutoff: float = 0.05, 
            counts1_name: str = CONDITION1_NAME, 
            counts2_name: str = CONDITION2_NAME, 
            design_factors: list[str] = [CONDITION_NAME], 
            csv_name: Optional[str] = None, 
            plot: bool = False,
            gene_list: Optional[list[str]] = None
            ) -> tuple[list[int], list[int], list[int], np.ndarray, pd.DataFrame, dict[str: list[str]]]:
    """Runs DESeq2 and returns significant genes and log fold changes
    Args:
        counts1 (numpy.ndarray): counts from condition 1
        counts2 (numpy.ndarray): counts from condition 2
        logfc_cutoff (float): log fold change cutoff
        pval_cutoff (float): p-value cutoff
        counts1_name (str): name of condition 1
        counts2_name (str): name of condition 2
        design_factors (list[str]): design factors
        csv_name (str): save results to csv if not None
        plot (bool): plot volcano plot or not
        gene_list (list[str]): gene list in column order
    Returns:
        up (list[int]): upregulared genes
        down (list[int]): downregulared genes
        same (list[int]): not differentially expressed genes
        log2foldchanges (numpy.ndarray): log fold changes
        dge_results_df (pd.DataFrame): DGE results
        genes (dict[str: list[str]]): dict of upregulared and downregulared genes
    """
    if counts1 is None or counts2 is None:
        return None, None, None, None, None
    deseq2_results_df = dge_stats_from_arrays(counts1, counts2, counts1_name, counts2_name, design_factors, csv_name = csv_name)
    up, down, same, log2foldchanges, _, colors = dge_genes_from_results_df(deseq2_results_df, logfc_cutoff, pval_cutoff)
    genes = {}
    if gene_list is not None:
        genes['up'] = [gene_list[i] for i in up]
        genes['down'] = [gene_list[i] for i in down]
    if plot:
        plt.scatter(deseq2_results_df['log2FoldChange'], -np.log10(deseq2_results_df['padj']), c = colors, edgecolors = 'k', linewidths = 0.05)
        plt.savefig('volcano-plot.png', bbox_inches = 'tight')
    return up, down, same, log2foldchanges, deseq2_results_df, genes

def dges_precision_recall(up1: list[int], 
                          down1: list[int], 
                          n_genes: int, 
                          up2: list[int], 
                          down2: list[int]
                          ) -> dict[str, float]:
    """Computes tpr, fpr and presicion
    Args:
        up1 (list[int]): list of ground truth upregulated genes
        down1 (list[int]): list of ground truth downregulated genes
        same1 (Union[int, list[int]]): list of ground truth not differentially expressed genes
        up2 (list[int]): list of predicted upregulated genes
        down2 (list[int]): list of predicted downregulated genes
    Returns:
        dict[str, float]: dictionary containing
            tpr_up: true positive rate (recall) on upregulated genes
            fpr_up: false positive rate on upregulated genes
            precision_up: precision on upregulated genes
            tpr_down: true positive rate (recall) on downregulated genes
            fpr_down: false positive rate on downregulated genes
            precision_down: precision on downregulated genes
    """
    tp_up_count, fp_up_count, tp_down_count, fp_down_count = 0, 0, 0, 0
    for gene in up2:
        if gene in up1:
            tp_up_count += 1
        else:
            fp_up_count += 1
    tpr_up = tp_up_count / len(up1) if len(up1) > 0 else 0
    fpr_up = fp_up_count / (n_genes - len(up1)) if (n_genes - len(up1)) > 0 else 0
    precision_up = tp_up_count / len(up2) if len(up2) > 0 else 0
    for gene in down2:
        if gene in down1:
            tp_down_count += 1
        else:
            fp_down_count += 1
    tpr_down = tp_down_count / len(down1) if len(down1) > 0 else 0
    fpr_down = fp_down_count / (n_genes - len(down1)) if (n_genes - len(down1)) > 0 else 0
    precision_down = tp_down_count / len(down2) if len(down2) > 0 else 0
    return_dict = {'tpr_up' : tpr_up, 'fpr_up' : fpr_up, 'precision_up' : precision_up, 'tpr_down' : tpr_down, 'fpr_down' : fpr_down, 'precision_down' : precision_down}
    return return_dict

def run_and_compare_DGEs(real_counts1: Optional[np.ndarray] = None, 
                         real_counts2: Optional[np.ndarray] = None, 
                         pred_counts1: Optional[np.ndarray] = None, 
                         pred_counts2: Optional[np.ndarray] = None, 
                         real_dge: Optional[pd.DataFrame] = None, 
                         pred_dge: Optional[pd.DataFrame] = None, 
                         gene_names: Optional[list[str]] = None, 
                         real_lfc_cutoff: float = 0.6, 
                         real_pval_cutoff: float = 0.05, 
                         pred_lfc_cutoff: float = 0.6, 
                         pred_pval_cutoff: float = 0.05, 
                         csv_name: Optional[str] = None
                         ) -> dict[str, float]:
    """Runs DGE and gene enrichment on real data and on predicted data and compares the results
    Args:
        real_counts1 (np.ndarray): real counts from condition 1
        real_counts2 (np.ndarray): real counts from condition 2
        pred_counts1 (np.ndarray): predicted counts from condition 1
        pred_counts2 (np.ndarray): predicted counts from condition 2
        real_dge (pd.DataFrame): DGE results on real data if available
        pred_dge (pd.DataFrame): DGE results on pred data if available
        gene_names (list[str]): list of gene names (as accepted by gseapy) whose counts are in the numpy arrays above in the same order
        real_lfc_cutoff (float): log fold change cutoff to use for DGE on real data (use value that makes sense biologically)
        real_pval_cutoff (float): p-value cutoff to use for DGE on real data (use value that makes sense biologically)
        pred_lfc_cutoff (float): log fold change cutoff to use for DGE on predicted data (use this value to adjust precision/recall)
        pred_pval_cutoff (float): p-value cutoff to use for DGE on predicted data (can also use this value to adjust precision/recall)
        csv_name (Optional[str]): save results to csv if not None
        fig_name (Optional[str]): plot log fold changes if not None
    Returns:
        dict[str, float]: contains output of dge_precision_recall, enrichment_precision_recall and correlations between LFCs
    """

    def dge_genes_(dge_results_df, counts_cond1, counts_cond2, logfc_cutoff, pval_cutoff, data_name = 'real', csv_name = None):
        if dge_results_df is not None:
            print('DGE results on', data_name, 'data available')
            up, down, same, lfcs, _, _ = dge_genes_from_results_df(dge_results_df, logfc_cutoff, pval_cutoff)
        elif counts_cond1 is not None and counts_cond2 is not None:
            print('Running DGE on', data_name, 'data...')
            csv_name = data_name + '-dge-' + csv_name + '.csv' if csv_name is not None else None
            up, down, same, lfcs, _, _ = run_dge(counts_cond1, counts_cond2, logfc_cutoff, pval_cutoff, csv_name = csv_name)
        else:
            raise ValueError('No DGE results or counts available for', data_name, 'data to find significant genes !')
        return up, down, same, lfcs 
    
    real_up, real_down, real_same, real_lfcs = dge_genes_(real_dge, real_counts1, real_counts2, real_lfc_cutoff, real_pval_cutoff, 'real', csv_name)
    pred_up, pred_down, _, pred_lfcs = dge_genes_(pred_dge, pred_counts1, pred_counts2, pred_lfc_cutoff, pred_pval_cutoff, 'pred', csv_name)
    results = compare_DGEs(real_up, real_down, real_same, real_lfcs, pred_up, pred_down, pred_lfcs, gene_names)   
    return results

mse = lambda x, y : np.nanmean(np.square(x - y))

def compare_DGEs(real_up: list[int], real_down: list[int], real_same: list[int], real_lfcs: np.ndarray, 
                 pred_up: list[int], pred_down: list[int], pred_lfcs: np.ndarray, 
                 gene_names: Optional[list[str]] = None) -> dict[str, float]:
    """Compares two DGE results
    Args:
        real_up (list[int]): list of ground truth upregulated genes
        real_down (list[int]): list of ground truth downregulated genes
        real_same (list[int]): list of ground truth not differentially expressed genes
        real_lfcs (np.ndarray): ground truth log fold changes
        pred_up (list[int]): list of predicted upregulated genes
        pred_down (list[int]): list of predicted downregulated genes
        pred_lfcs (np.ndarray): predicted log fold changes
        gene_names (list[str]): list of gene names (as accepted by gseapy) whose counts are in the numpy arrays above in the same order
    Returns
        dict[str, float]: contains output of dge_precision_recall, enrichment_precision_recall and distances between LFCs
    """
    
    dict1 = dges_precision_recall(real_up, real_down, real_same, pred_up, pred_down)
    dict2, dict3, dict4 = {}, {}, {}
    
    try:
        spearman, pearson = stats.spearmanr(real_lfcs, pred_lfcs), stats.pearsonr(real_lfcs, pred_lfcs)
        dict2 = {'spearman_corr' : spearman.statistic, 'spearman_pval' : spearman.pvalue, 'pearson_corr' : pearson.statistic, 'pearson_pval' : pearson.pvalue}
    except ValueError:
        print("[X] Could not compute spearman and pearson stats on DGE LFC.")
        
    dict2['lfc_mse'] = mse(real_lfcs, pred_lfcs)
        
    if gene_names is not None and len(real_up) > 0 and len(pred_up) > 0:
        real_enrichment_up = gene_enrichments(extract_sublist(gene_names, real_up))
        pred_enrichment_up = gene_enrichments(extract_sublist(gene_names, pred_up)) 
        dict3 = enrichment_precision_recall(real_enrichment_up, pred_enrichment_up, 'up')

    if gene_names is not None and len(real_down) > 0 and len(pred_down) > 0:
        real_enrichment_down = gene_enrichments(extract_sublist(gene_names, real_down))
        pred_enrichment_down = gene_enrichments(extract_sublist(gene_names, pred_down)) 
        dict4 = enrichment_precision_recall(real_enrichment_down, pred_enrichment_down, 'down')
        
    return {**dict1, **dict2, **dict3, **dict4}
    

def gene_ids_to_names(gene_ids: list[str]) -> list[str]:
    """Gene IDs to names via Biomart from gseapy
    Args:
        gene_ids (list[str]): list of gene ids
    Returns:
        gene_names (list[str]): list of gene names
    """
    
    bm = gp.Biomart()
    queries = {'ensembl_gene_id': gene_ids}
    gene_names = bm.query(dataset = 'hsapiens_gene_ensembl', attributes = ['ensembl_gene_id', 'external_gene_name', 'entrezgene_id', 'go_id'], filters = queries)
    print(gene_names)
    gene_names = [x for x in gene_names['external_gene_name'].unique().tolist() if type(x) == str]
    print('gene ids to gene names', len(gene_names), '/', len(gene_ids))
    return gene_names

def gene_enrichments(gene_names: list[str], gene_sets: list[str] = GENE_SETS) -> dict[str, list[str]]:
    """Gene enrichment on gene_names (list) using all gene sets
    Args:
        gene_names (list[str]): list of gene names
        gene_sets (list[str]): list of gene sets to use
    Returns:
        dict[str, list[str]]: keys are gene sets, values are lists of pathways
    """
    
    gene_enrichment_terms = {}
    for gene_set in gene_sets:
        print('\nGene enrichment using', gene_set)
        enr_up = gp.enrichr(gene_names, gene_sets = gene_set, outdir = None, cutoff = 0.2)
        enr_up.res2d.Term = enr_up.res2d.Term.str.split(" \(GO").str[0]
        gene_enrichment_terms[gene_set] = enr_up.res2d['Term'].tolist()
    return gene_enrichment_terms

def enrichment_precision_recall(gene_enrichment_terms_1: dict[str, list[str]], 
                                gene_enrichment_terms_2: dict[str, list[str]],
                                comment: str = '') -> dict[str, float]:
    """Computes precision and recall for gene enrichment pathways for different gene sets
    Args:
        gene_enrichment_terms_1 (dict[str, list[str]]): the values are lists of pathways and the keys are gene sets
        gene_enrichment_terms_2 (dict[str, list[str]]): the values are lists of pathways and the keys are gene sets
        comment(str): add to dict keys in output
    Returns:
        dict[str, float]: the values are precision and recall for each gene set
    """
    precision_recall = {}
    for gene_set, enrichment_terms in gene_enrichment_terms_2.items():
        count_true = 0
        for term in enrichment_terms:
            if term in gene_enrichment_terms_1[gene_set]:
                count_true += 1
        precision = count_true / len(enrichment_terms) if len(enrichment_terms) > 0 else 0
        recall = count_true / len(gene_enrichment_terms_1[gene_set]) if len(gene_enrichment_terms_1[gene_set]) > 0 else 0
        precision_recall['enrch_' + comment + '_recall_' + gene_set] = recall
        precision_recall['enrch_' + comment + '_precision_' + gene_set] = precision
    return precision_recall

def get_dge_results_df(dge_file_name: Optional[str] = None, 
                       samples_cond1: Optional[np.ndarray] = None, 
                       samples_cond2: Optional[np.ndarray] = None, 
                       save_name: Optional[str] = None) -> pd.DataFrame:
    """Get DGE results dataframe
    Args:
        dge_file_name (Optional[str]): DGE results csv file name
        samples_cond1 (Optional[np.ndarray]): counts with condition 1
        samples_cond2 (Optional[np.ndarray]): counts with condition 2
        save_name (Optional[str]): save name for DGE results
    Returns:
        pd.DataFrame: DGE results dataframe
    """
    if dge_file_name is None:
        if samples_cond1 is not None and samples_cond2 is not None:
            dge_results_df = dge_stats_from_arrays(samples_cond1, samples_cond2)
            if save_name is not None:
                dge_results_df.to_csv(save_name)
        else:
            raise ValueError('Provide DGE file name or samples to run DGE !')
    else:
        dge_results_df = pd.read_csv(dge_file_name)
    return dge_results_df  