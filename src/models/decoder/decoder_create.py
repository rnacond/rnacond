from src.models.decoder.decoder_processor import DecoderProcessor, DecoderDatasetConfig
from src.models.decoder.decoder_model import Decoder, DecoderConfig
from src.dataops.dataloaders.data_processor import create_dataloaders, DataLoaderConfig 
from src.lightning_modules.generative_lightning_module import GenerativeLightningModule, GenerativeLightningModuleConfig
from src.dataops.dataloaders.gene_dataset import compute_block_size
from src.utils.utils import get_attributes
import torch
from typing import Optional
from src.dataops.data_utils import extract
import pickle
import anndata as ad
from typing import Any
from torch.utils.data import DataLoader
from src.modules.bert import BERT, get_bert_embeddings
from src.dataops.tokenizers.bert_tokenizer import get_bert_tokenizer
import pandas as pd

def create_decoder_for_training(adata: ad.AnnData, configs: dict[str, dict[str, Any]]) -> tuple[Decoder, GenerativeLightningModule, DataLoader, DataLoader]:
    """Creates model and lightning model and dataloader

    Args:
        adata (ad.AnnData): data
        configs (dict[str, dict[str, Any]]): must contain 'model_config', 'dataset_config', 'dataloader_config', 'lightning_config'
        
    Returns:
        (Decoder): model
        (GenerativeLightningModule): lightning model
        (DataLoader): train dataloader
        (DataLoader): valid dataloader
    """

    model_config = configs['model_config']
    dataset_config = configs['dataset_config']
    dataloader_config = configs['dataloader_config']
    lightning_config = configs['lightning_config']
    prompts = configs['experiment_config']['prompts']
    tokenizer_path = configs['experiment_config']['tokenizer_path']
    gene_list = adata.var_names.tolist()
    gene_symbols = list(adata.var['gene_symbols'].astype(str).values) if 'gene_symbols' in adata.var.keys() else None
    
    assert model_config['embedding_dim'] % model_config['num_heads'] == 0, 'embedding_dim should be divisible by num_heads'
    
    if tokenizer_path is not None: 
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
                print('Loaded IBD tokenizer')
        except FileNotFoundError:
            tokenizer = None
            print('Failed to load tokenizer')
    else:
        tokenizer = None
        
    dataset_config = DecoderDatasetConfig(**dataset_config)
    dataloader_config = DataLoaderConfig(**dataloader_config)
    data_processor = DecoderProcessor(dataset_config, dataloader_config, adata, tokenizer)
    gene_tokenizer = data_processor.tokenizers['gene_tokenizer']
    train_dataloader, valid_dataloader = create_dataloaders(data_processor, adata)
    
    if model_config['pretrained_embeddings']:
        metadata_tokens = gene_tokenizer.metadata_tokens
        bert, bert_tokenizer = BERT(), get_bert_tokenizer()
        pretrained_metadata_embeddings = get_bert_embeddings(bert, bert_tokenizer, metadata_tokens)
    else:
        pretrained_metadata_embeddings = None
    
    vocab_size = gene_tokenizer.get_vocab_size()
    block_size = compute_block_size(dataset_config, True)
    
    model_config_from_dataset = get_attributes(dataset_config, ['metadata_names', 'num_total_counts', 'sample', 'shuffle_metadata'])
    model_config = DecoderConfig(pretrained_metadata_embeddings = pretrained_metadata_embeddings, gene_list = gene_list, gene_symbols = gene_symbols, vocab_size = vocab_size, block_size = block_size, **model_config_from_dataset, **model_config)
    model = Decoder(model_config, gene_tokenizer)
    
    num_samples = 200
    generation_val_data = [model.raw_counts_to_model_counts(extract(adata, context, num_samples)) for context in prompts]
    real_dge = pd.read_csv(configs['experiment_config']['dge_path'])  
    pseudobulk_dge = 'pb' in configs['experiment_config']['dge_path'] 
    
    lightning_config = GenerativeLightningModuleConfig(**lightning_config, pseudobulk_dge = pseudobulk_dge)
    lightning_model = GenerativeLightningModule(model, lightning_config, model_config, gene_tokenizer, prompts, generation_val_data, real_dge) 

    return model, lightning_model, train_dataloader, valid_dataloader


def load_decoder_model(checkpoint_path: str, gene_tokenizer_path: Optional[str] = None) -> Decoder:
    """Loads model from checkpoint

    Args:
        checkpoint_path (str): path to checkpoint
        gene_tokenizer_path (Oprional[str]): path to gene tokenizer, a pickle file

    Returns:
        Decoder: model
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, weights_only = False, map_location = device)
    config = ckpt['hyper_parameters']['model_config']
    if not hasattr(config, 'gene_symbols'):
        config.gene_symbols = None
    with open(gene_tokenizer_path, 'rb') as inp:
        gene_tokenizer = pickle.load(inp)
    model = Decoder(config, gene_tokenizer)
    state_dict = {key.removeprefix('model.'): value for key, value in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

