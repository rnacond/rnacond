from __future__ import annotations
import fire
import torch
import numpy as np 
from lightning.pytorch import seed_everything
from src.utils.parse import parse
import anndata as ad
import lightning as L
from src.dataops.preprocess import preprocess_adata
from lightning.pytorch.callbacks import ModelCheckpoint
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from src.models.create_models import create_for_training    
from src.metrics.generation_metrics import evaluate_generation, GenerationContext
from lightning.pytorch.loggers import WandbLogger
import re

if __name__ == "__main__":
    
    colorama_init(autoreset = True)
    print(f"\n{Fore.MAGENTA}***************** TRAINING EXP ******************* {Style.RESET_ALL}\n")
    
    configs = fire.Fire(parse)
    
    model_name = configs['experiment_config']['model']
    metadata_names = configs['experiment_config']['metadata_names'] 
    data_path = configs['experiment_config']['data']
    seed = configs['experiment_config']['seed']
    checkpoint = configs['experiment_config']['checkpoint']
    
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_everything(seed, workers = True)
    
    adata = ad.read_h5ad(data_path)
    adata = preprocess_adata(adata, metadata_names, configs['preprocess_config'])
    print(f"\n{Fore.MAGENTA}***************** ANNDATA ******************* {Style.RESET_ALL}\n", adata, f"\n{Fore.MAGENTA}***************** ANNDATA ******************* {Style.RESET_ALL}\n")
    
    model, lightning_model, train_dataloader, valid_dataloader = create_for_training(model_name, adata, configs)
    
    
    
    if torch.cuda.is_available():
        
        wandb_logger = WandbLogger(project = model_name + '_' + re.split('[/.]', data_path)[2], log_model = 'all')
        
        checkpoint_callback = ModelCheckpoint(dirpath = "./checkpoints",  # Specify the directory to save checkpoints
            filename = "scDecoder-{epoch:02d}-{valid_loss:.4f}",  # Customize the filename
            monitor = 'valid_loss',  # Monitor the validation loss
            mode = "min",  # Save the best model based on the minimum loss
            save_top_k = 3, # Save the top 3 best models
            save_last = True # Save the last checkpoint
        )
        
        trainer = L.Trainer(**configs['trainer_config'], logger = wandb_logger, callbacks = [checkpoint_callback])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision('high')
        print(f"\n{Fore.MAGENTA}***************** TRAINING START ******************* {Style.RESET_ALL}\n")
        trainer.fit(model = lightning_model, train_dataloaders = train_dataloader, val_dataloaders = valid_dataloader, ckpt_path = 'last')
        
        if model_name in ['decoder', 'encoder_decoder', 'diffusion']:
            print(f"\n{Fore.MAGENTA}***************** EMD EVAL ******************* {Style.RESET_ALL}\n")
            model.set_device(device)
            model.eval()
            context = GenerationContext(metadata_context = {'disease' : 'healthy', 'organ' : 'heart'}) if 'toy' in data_path else GenerationContext()
            emd = evaluate_generation(context, model, adata, 5, configs['dataloader_config']['batch_size'], 60000)
            print('EMD:', emd)
            
            
  