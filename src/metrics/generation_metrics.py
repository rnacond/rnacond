from __future__ import annotations
import numpy as np 
import ot
from typing import Literal, Union
from src.dataops.data_utils import extract
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import anndata as ad
from src.modules.base_generative_model import GenerationContext, GenerativeModel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def evaluate_generation(context: GenerationContext, 
                        model: GenerativeModel, 
                        adata: ad.AnnData,
                        num_samples: int = 8, 
                        batch_size: int = 8, 
                        max_new_tokens: int = 2000
                        ) -> float:
    
    """Extracts and generates data based on conditions and computes distances

    Args:
        context (GenerationContext): context
        model (nn.Module): model to generate from
        adata (ad.AnnData): data to extract from
        num_samples (int): number of samples to generate/extract
        batch_size (int): batch size
        max_new_tokens (int): maximum number of new tokens to generate
        
    Returns:
        float: emd distance
    """
    
    assert model.gene_list == adata.var_names.tolist(), 'gene lists should match between model and data'
    extracted_anndata = extract(adata, context.metadata_context, num_samples)
    if len(extracted_anndata) == 0:
        print('[!] No samples match these conditions. Cannot compute EMD.')
        return np.nan
    extracted = model.raw_counts_to_model_counts(extracted_anndata)
    generated = model.generate(context = context, num_samples = extracted.shape[0], batch_size = batch_size, max_new_tokens = max_new_tokens) 
    generated = generated.X if isinstance(generated, ad.AnnData) else generated
    return ot_distances(extracted, generated)  

def prepare_for_ot(samples1: np.ndarray, samples2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates distance matrix between uniform weights for two empirical distributions.
    
    Args:
        samples1 (np.ndarray): first empirical data set
        samples2 (np.ndarray): second empirical data set
        
    Returns:
        samples1_weights (np.ndarray): uniform weights
        samples2_weights (np.ndarray): uniform weights
        distance_matrix (np.ndarray): matrix of distances between data points in samples1 and samples2
    """
    uniform_weights = lambda n : np.ones(n) / n
    num_samples1, num_samples2 = samples1.shape[0], samples2.shape[0]
    distance_matrix = np.zeros((num_samples1, num_samples2))
    for i in range(num_samples1):
        for j in range(num_samples2):
            distance_matrix[i,j] = np.linalg.norm(samples1[i] - samples2[j], 2)
    samples1_weights, samples2_weights = uniform_weights(num_samples1), uniform_weights(num_samples2)
    return samples1_weights, samples2_weights, distance_matrix

def ot_distances(samples1: np.ndarray, 
                 samples2: np.ndarray, 
                 which: Literal['both', 'emd', 'sinkhorn'] = 'emd', 
                 sinkhorn_reg = 0.1
                 ) -> Union[dict[str, float], float]:
    """Computes and prints Sinkhorn and EMD distances between samples1 and samples2.
    
    Args:
        samples1 (np.ndarray): first empirical data set
        samples2 (np.ndarray): second empirical data set
        which (Literal['all', 'emd', 'sinkhorn']): which distance(s) to compute, default to 'emd'
        sinkhorn_reg (float): regularization weight for sinkhorn
        
    Returns:
        Union[dict[str, float], float]: a dictionary with both distances or just one as a float
    """
    
    frobenius_inner_product = lambda a, b : np.sum(np.multiply(a, b))
    samples1_weights, samples2_weights, distance_matrix = prepare_for_ot(samples1, samples2)
    res = dict()
    if which == 'both' or which == 'sinkhorn':
        transport_matrix = ot.sinkhorn(samples1_weights, samples2_weights, distance_matrix, sinkhorn_reg)
        dist = frobenius_inner_product(transport_matrix, distance_matrix)
        res['sinkhorn'] = dist
    if which == 'both' or which == 'emd':
        transport_matrix = ot.emd(samples1_weights, samples2_weights, distance_matrix)
        dist = frobenius_inner_product(transport_matrix, distance_matrix)   
        res['emd'] = dist
    return res if which == 'both' else dist

def prepare_for_classification(samples1: np.ndarray, samples2: np.ndarray, test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Prepeares data for classification algorithms
    
    Args:
        samples1 (np.ndarray): data (e.g. real)
        samples2 (np.ndarray): data (e.g. fake)
        test (bool): train-test split or not
        

    Returns:
        np.ndarray: X (features)
        np.ndarray: Y (labels)
    """
    
    X = np.concatenate((samples1, samples2))
    Y = np.concatenate((np.full((samples1.shape[0], ), 0), np.full((samples2.shape[0], ), 1)))
    if not test:
        return X, Y
    else:
        np.random.shuffle(X)
        np.random.shuffle(Y)
        n_train = int(0.8 * X.shape[0])
        X_train, Y_train, X_test, Y_test = X[:n_train,:], Y[:n_train], X[n_train:,:], Y[n_train:]
        return X_train, Y_train, X_test, Y_test
    
@ignore_warnings(category = ConvergenceWarning)
def lr_accuracy(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """Computes separability of real data (samples1) from generated data (samples2) using logistic regression.
    
    Args:
        samples1 (np.ndarray): data (e.g. real)
        samples2 (np.ndarray): data (e.g. fake)

    Returns:
        float: logistic regression accuracy
    """
    X, Y = prepare_for_classification(samples1, samples2)
    LR = LogisticRegressionCV().fit(X, Y)
    return LR.score(X, Y)

def rf_accuracy(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """Computes separability of real data (samples1) from generated data (samples2) using random forest.
    
    Args:
        samples1 (np.ndarray): data (e.g. real)
        samples2 (np.ndarray): data (e.g. fake)

    Returns:
        float: random forest accuracy
    """
    
    X, Y = prepare_for_classification(samples1, samples2)
    RF = RandomForestClassifier().fit(X, Y)
    return RF.score(X, Y)


def distances(real_counts : np.ndarray, fake_counts: np.ndarray) -> dict[str, float]:
    """Returns EMD, LR and RF accuracy and scDesign metrics

    Args:
        real_counts (np.ndarray): real counts
        fake_counts (np.ndarray): fake counts

    Returns:
        dict[str, float]: dictionary with metrics
    """
    
    metrics = dict()
    metrics['EMD'] = ot_distances(real_counts, fake_counts)
    metrics['LR ACC'] = lr_accuracy(real_counts, fake_counts)
    metrics['RF ACC'] = rf_accuracy(real_counts, fake_counts)
    return {**metrics, **scdesign_metrics(real_counts, fake_counts)}

def test_classification_on_fake_data(real_array1 : np.ndarray, 
                                     real_array2 : np.ndarray, 
                                     fake_array1 : np.ndarray, 
                                     fake_array2 : np.ndarray
                                     ) -> dict[str, float]:
    """Tests training a LR on fake data and testing on real data
    
    Args:
        real_array1 (np.ndarray) 
        real_array2 (np.ndarray)
        fake_array1 (np.ndarray)
        fake_array2 (np.ndarray)

    Returns:
        dict[str, float]: results
    """
    
    results = {}
    X_train_real, Y_train_real, X_test_real, Y_test_real = prepare_for_classification(real_array1, real_array2, True)
    pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter = 900))
    pipe.fit(X_train_real, Y_train_real)
    results['LR_real_train_acc'] = pipe.score(X_train_real, Y_train_real)
    results['LR_real_test_acc'] = pipe.score(X_test_real, Y_test_real)

    X_train_fake, Y_train_fake, _, _ = prepare_for_classification(fake_array1, fake_array2, True)
    pipe.fit(X_train_fake, Y_train_fake)
    results['LR_fake_train_acc'] = pipe.score(X_train_fake, Y_train_fake)
    results['LR_train_on_fake_test_on_real_acc'] = pipe.score(X_test_real, Y_test_real)
    
    print(results)
    return results

    

cv =  lambda x : np.std(x) / np.mean(x)
mse = lambda x, y : (np.square(x - y)).mean(axis = None).item()

def pearsonr(samples: np.ndarray) -> np.ndarray:
    """Computes a matrix of Pearson correlations between gene-pairs. Slow, give a subset of genes.
    
    Args:
        samples (np.ndarray): samples (cells x counts)
        
    Returns:
        (np.ndarray): Pearson correlation for gene-pairs
    """
    
    pearsonr_matrix = np.zeros((samples.shape[1], samples.shape[1]))
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            pearsonr_matrix[i,j] = stats.pearsonr(samples[:, i], samples[:, j]).statistic
    return pearsonr_matrix

def kendallt(samples: np.ndarray) -> np.ndarray:
    """Computes a matrix of Kendall's tau between gene-pairs. Slow, give a subset of genes.
    
    Args:
        samples (np.ndarray): samples (cells x counts)
        
    Returns:
        (np.ndarray): Kendall's tau for gene-pairs
    """
    
    kendallt_matrix = np.zeros((samples.shape[1], samples.shape[1]))
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            kendallt_matrix[i,j] = stats.kendalltau(samples[:, i], samples[:, j]).statistic
    return kendallt_matrix


def scdesign_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Computes the different statistics in the scDesign2 paper.
    
    Args:
        samples (np.ndarray): samples (cells x counts)
        
    Returns:
        dict[str, np.ndarray]: dictionary with statistic name and np.ndarray containing the statistic 
    """
    
    stats_dict = {}
    stats_dict['mean_genes'] = np.mean(samples, axis = 0)
    stats_dict['var_genes'] = np.var(samples, axis = 0)
    #stats_dict['cv1_genes'] = np.apply_along_axis(cv, axis = 0, arr = samples)
    #stats_dict['cv2_genes'] = stats.variation(samples, axis = 0)
    stats_dict['zp_genes'] = (samples == 0).sum(0) / samples.shape[0]
    stats_dict['skew_genes'] = stats.skew(samples, axis = 0)
    stats_dict['kurt_genes'] = stats.kurtosis(samples, axis = 0)
    # stats_dict['zp_cells'] = (samples == 0).sum(1) / samples.shape[1] # mse does not make sense for this
    # stats_dict['pearsonr_matrix'] = pearsonr(samples) # slow
    # stats_dict['kendallt_matrix'] = kendallt(samples) # slow
    return stats_dict

def scdesign_metrics(samples1: np.ndarray, samples2: np.ndarray) -> dict[str, float]:
    """MSEs between scdesign statistics on samples1 and samples2. Does not make sense for cell zero proportion.
    
    Args:
        samples1 (np.ndarray): samples (cells x counts)
        samples2 (np.ndarray): samples (cells x counts)
        
    Returns:
        dict[str, float]: dictionary containing the MSEs
    """
    
    print('\n[X] Computing ScDesign metrics...')
    samples1_stats_dict = scdesign_stats(samples1)
    samples2_stats_dict = scdesign_stats(samples2)
    return dict([('mse_' + stat_name, mse(samples1_stats_dict[stat_name], samples2_stats_dict[stat_name])) for stat_name in samples1_stats_dict])
    




