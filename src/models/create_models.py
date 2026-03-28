from src.models.clip.clip_create import create_clip_for_training, load_clip_model
from src.models.decoder.decoder_create import create_decoder_for_training, load_decoder_model
from src.models.diffusion.diffusion_create import create_diffusion_for_training, load_diffusion_model
from src.models.encoder.encoder_create import create_encoder_for_training, load_encoder_model
from src.models.encoder_decoder.encoder_decoder_create import create_encoder_decoder_for_training, load_encoder_decoder_model
import anndata as ad
from typing import Any, Literal, Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from src.lightning_modules.lightning_module import LightningModule


def create_for_training(model_name: Literal['decoder', 'encoder', 'clip', 'encoder_decoder', 'diffusion'],
                        adata: ad.AnnData, 
                        configs: dict[str, dict[str, Any]]
                        ) -> tuple[nn.Module, LightningModule, DataLoader, DataLoader]:
    """Creates model and lightning model and dataloaders for training
    
    Args:
        model_name (Literal['decoder', 'encoder', 'clip', 'encoder_decoder', 'diffusion']): model name
        adata (ad.AnnData): data
        configs (dict[str, dict[str, Any]]): must contain 'model_config', 'dataset_config', 'dataloader_config', 'lightning_config'
        
    Returns:
        (nn.Module): model
        (LightningModule): lightning model
        (DataLoader): train dataloader
        (DataLoader): valid dataloader
    """
    
    if model_name == 'decoder':
        return create_decoder_for_training(adata, configs)
    elif model_name == 'encoder':
        return create_encoder_for_training(adata, configs)
    elif model_name == 'clip':
        return create_clip_for_training(adata, configs)
    elif model_name == 'encoder_decoder':
        return create_encoder_decoder_for_training(adata, configs)
    elif model_name == 'diffusion':
        return create_diffusion_for_training(adata, configs)
    else:
        raise ValueError(f"Model {model_name} not supported")


def load_model(model_name: Literal['decoder', 'encoder', 'clip', 'encoder_decoder', 'diffusion'],
               checkpoint_path: str,
               gene_tokenizer_path: Optional[str] = None
               ) -> nn.Module:
    """Loads a model from a checkpoint

    Args:
        model_name (Literal['decoder', 'encoder', 'clip', 'encoder_decoder', 'diffusion']): model_name
        checkpoint_path (str): checkpoint path
        gene_tokenizer_path (str, optional): gene_tokenizer_path. Defaults to None.

    Returns:
        nn.Module: model with weights
    """
    
    if model_name in ['decoder', 'encoder'] and gene_tokenizer_path is None:
        raise ValueError('You must provide the tokenizer the model was trained with')
        
    if model_name == 'decoder':
        return load_decoder_model(checkpoint_path, gene_tokenizer_path)
    elif model_name == 'encoder':
        return load_encoder_model(checkpoint_path, gene_tokenizer_path)
    elif model_name == 'clip':
        return load_clip_model(checkpoint_path) 
    elif model_name == 'encoder_decoder':
        return load_encoder_decoder_model(checkpoint_path) 
    elif model_name == 'diffusion':
        return load_diffusion_model(checkpoint_path)
    else:
        raise ValueError(f"Model {model_name} not supported")