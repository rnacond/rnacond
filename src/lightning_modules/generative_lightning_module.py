
from src.lightning_modules.lightning_module import LightningModule, LightningModuleConfig
from torch import Tensor
from src.metrics.generation_metrics import lr_accuracy, ot_distances, scdesign_metrics, pearsonr, rf_accuracy
from src.metrics.dge import run_and_compare_DGEs
import numpy as np
from src.modules.base_generative_model import GenerativeModel, GenerationContext
from pydantic import BaseModel
from src.dataops.pseudobulk import create_pseudobulk
import anndata as ad
from typing import Optional
from src.metrics.plots import plot_rgb_correlations
import pandas as pd
from src.metrics.dge import GENE_SETS 
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer
from src.utils.utils import to_array


class GenerativeLightningModuleConfig(LightningModuleConfig):
    verbose: bool = False
    pseudobulk_dge: bool = False

class GenerativeLightningModule(LightningModule):
    def __init__(self, 
                 model: GenerativeModel, 
                 config: GenerativeLightningModuleConfig, 
                 model_config: Optional[BaseModel] = None,
                 tokenizer: Optional[GeneTokenizer] = None,
                 validation_prompts: list[dict[str, str]] = [{}],
                 generation_val_data: Optional[list[np.ndarray]] = None,
                 real_dge: Optional[pd.DataFrame] = None
                 ) -> None:
        """Constructor

        Args:
            model (GenerativeModel): model
            config (GenerativeLightningModuleConfig): config
            model_config (Optional[Union[dict[str, Any], BaseModel]]): model config if you want it saved in checkpoint
            tokenizer (Optional[GeneTokenizer]): tokenizer
            validation_prompts (list[dict[str, str]]): generation prompts for validation
            generation_val_data (Optional[list[np.ndarray]]): val data corresponding to the prompts
            real_dge (Optional[pd.DataFrame]): ground truth dge between last two prompts
        """
        
        super().__init__(model, config, model_config, tokenizer)
        self.model = model
        self.validation_prompts = validation_prompts
        self.generation_val_data = generation_val_data if generation_val_data is not None else [None for _ in validation_prompts]
        self.real_dge = real_dge
        self.n_dge_genes = len(real_dge)
        self.gene_ids = self.model.gene_list[:self.n_dge_genes]
        self.gene_symbols = self.model.gene_symbols[:self.n_dge_genes] if self.model.gene_symbols is not None else None
        
    def _generate(self, 
                  context: GenerationContext, 
                  num_samples: int = 5, 
                  batch_size: int = 5, 
                  run_on = None) -> ad.AnnData:
        """Generates samples

        Args:
            context (GenerationContext): context to start generate from
            num_samples (int): number of samples to generate
            batch_size (int): batch size
            run_on (Union[torch.device, Tensor]): device to run on

        Returns:
            ad.AnnData: generated data
        """
        generated = self.model.generate(context, num_samples, batch_size, verbose = self.config.verbose, run_on = run_on)
        return generated
    
    def _plot(self, 
              counts: np.ndarray, 
              plot_counts: bool = True, 
              plot_correlations: bool = True, 
              log_message: str = '',
              n_genes: int = 10
              ) -> None:
        """Plots generated counts and correlations as greyscale images in wandb

        Args:
            counts (np.ndarray): counts (cells x genes)
            plot_counts (bool, optional): plot counts. Defaults to True.
            plot_correlations (bool, optional): plot correlations. Defaults to True.
            n_genes (int, optional): half the number of genes to consider. Default to 10.
            log_message (str, optional): message to add to log in wandb. Defaults to ''.
            
        Returns:
            None
        """
        
        if not plot_counts and not plot_correlations:
            return
        assert n_genes > 1
        assert n_genes % 2 == 0
        images = np.hstack((counts[:, :n_genes], counts[:, -n_genes:]))
        if plot_correlations:
            try:
                correlations = [plot_rgb_correlations(np.nan_to_num(pearsonr(images)))]
                self.logger.log_image(key = "Correlations " + log_message, images = correlations)
            except ValueError:
                print('[X] Did not compute correlations for this step, nan or inf encountered.')
        if plot_counts:
            images = images.reshape((images.shape[0], 4, int(n_genes / 2)))
            images = [images[i] for i in range(images.shape[0])]
            self.logger.log_image(key = "Generated " + log_message, images = images)
        
            
    def _metrics(self, 
                 x: np.ndarray, 
                 generated_counts: np.ndarray, 
                 compute_ot_distances: bool = True, 
                 compute_lr: bool = True,
                 compute_rf: bool = True,
                 compute_scdesign_metrics: bool = True,
                 log: bool = True,
                 log_message: str = ''
                 ) -> None:
        """Logs generation metrics (distances between generated and data)

        Args:
            x (np.ndarray): real counts
            generated_counts (np.ndarray): generated
            compute_ot_distances (bool, optional): compute optimal transport distances. Defaults to True.
            compute_lr (bool, optional): compute LR accuracy or not. Defaults to True.
            compute_rf (bool, optional): compute RF accuracy or not. Defaults to True.
            compute_scdesign_metrics (bool, optional): compute scdesign metrics. Defaults to False.
            log_or_return (bool, optional): log metrics
            log_message (str, optional): message to add to log in wandb. Defaults to ''.
            
        Returns:
            None
        """
        
        batch_size = x.shape[0]
        
        if compute_ot_distances:
            print('Computing EMD for prompt', log_message, '...')    
            emd = ot_distances(generated_counts, x)
            if log:
                self.log("EMD " + log_message, emd, prog_bar = True, sync_dist = True, batch_size = batch_size)
                
        if compute_lr:
            print('Fitting LR for prompt', log_message, '...')    
            lr = lr_accuracy(generated_counts, x)
            if log:
                self.log("LR ACC " + log_message, lr, prog_bar = True, sync_dist = True, batch_size = batch_size)
                
        if compute_rf:
            print('Fitting RF for prompt', log_message, '...')    
            rf = rf_accuracy(generated_counts, x)
            if log:
                self.log("RF ACC " + log_message, rf, prog_bar = True, sync_dist = True, batch_size = batch_size)
                
        if compute_scdesign_metrics:
            print('Computing scDesign metrics for prompt', log_message, '...')    
            scdesign_metrics_dict = scdesign_metrics(generated_counts, x)
            if log:
                self.log("MSE MEAN GENES " + log_message, scdesign_metrics_dict['mse_mean_genes'], prog_bar = True, sync_dist = True, batch_size = batch_size)
                self.log("MSE VAR GENES " + log_message, scdesign_metrics_dict['mse_var_genes'], prog_bar = True, sync_dist = True, batch_size = batch_size)
        
        
    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """training step

        Args:
            batch (dict[str, Tensor]): input and labels
            batch_idx (int): batch index
            
        Returns:
            Tensor: loss
        """
        
        self._loss(batch, 'valid')
        
        if batch_idx == 0:
            
            x, batch_size = batch['x'], batch['x'].size(0)
            generated_adatas = []
            for context, real_counts in zip(self.validation_prompts, self.generation_val_data):
                
                prompt = GenerationContext(metadata_context = context)
                log_message = prompt.to_str()
                real_counts = self.model.batch_to_counts(x, verbose = False) if real_counts is None else real_counts
                num_samples = real_counts.shape[0]
                print("Generating from prompt", log_message, '...')    
                generated_adata = self._generate(prompt, num_samples, batch_size, run_on = x)
                generated_counts = generated_adata.X
                print("Plotting for prompt", log_message, '...')    
                self._plot(generated_counts, log_message = log_message)
                self._metrics(real_counts, generated_counts, log = True, log_message = log_message)
                generated_adatas.append(generated_adata) 
                
            if len(generated_adatas) > 1:
                
                if self.config.pseudobulk_dge:
                    generated_counts = [to_array(create_pseudobulk(generated_adata, 'sample').X)[:, :self.n_dge_genes] for generated_adata in generated_adatas]
                else:
                    generated_counts = [to_array(generated_adata.X)[:, :self.n_dge_genes] for generated_adata in generated_adatas]
                
                try:
                    dge = run_and_compare_DGEs(pred_counts1 = generated_counts[-1], pred_counts2 = generated_counts[-2], real_dge = self.real_dge, gene_names = self.gene_symbols)
                    
                    self.log("Recall UP ", dge['tpr_up'], prog_bar = True, sync_dist = True, batch_size = batch_size)                
                    self.log("Precision UP ", dge['precision_up'], prog_bar = True, sync_dist = True, batch_size = batch_size)                
                    self.log("FPR UP ", dge['fpr_up'], prog_bar = True, sync_dist = True, batch_size = batch_size)  
                    self.log("Recall DOWN ", dge['tpr_down'], prog_bar = True, sync_dist = True, batch_size = batch_size)                
                    self.log("Precision DOWN ", dge['precision_down'], prog_bar = True, sync_dist = True, batch_size = batch_size)                
                    self.log("FPR DOWN ", dge['fpr_down'], prog_bar = True, sync_dist = True, batch_size = batch_size)               
                    self.log("LFC MSE ", dge['lfc_mse'], prog_bar = True, sync_dist = True, batch_size = batch_size) 
                    for gene_set in GENE_SETS:
                        for up_down in ['up', 'down']:
                            for metric in ['_precision_', '_recall_']:
                                try:
                                    name = 'enrch_' + up_down + metric + gene_set
                                    self.log(name, dge[name], prog_bar = True, sync_dist = True, batch_size = batch_size) 
                                except KeyError:
                                    pass
                except ValueError:
                    print('[!] Error doing pred DGE. Skipping.')
