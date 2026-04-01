import lightning as L
from torch import nn, Tensor
import torch
from typing import Literal, Union
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from pydantic import BaseModel
from typing import Optional
from src.dataops.tokenizers.gene_tokenizer import GeneTokenizer


class LightningModuleConfig(BaseModel):
    optimizer: Literal['adamw', 'sgd'] = 'adamw'
    scheduler: Literal['cosinelr', 'steplr', 'none'] = 'steplr'
    learning_rate: float = 3e-4
    num_epochs: int = 100
    step_size: int = 30
    
    


class LightningModule(L.LightningModule):
    def __init__(self, 
                 model: nn.Module, 
                 config: LightningModuleConfig,
                 model_config: Optional[BaseModel] = None,
                 tokenizer: Optional[GeneTokenizer] = None,
                 ) -> None:
        """Constructor

        Args:
            model (nn.Module): model
            config (Union[dict[str, Any], LightningModuleConfig]): config
            model_config (Optional[Union[dict[str, Any], BaseModel]]): model config if you want it saved in checkpoint
            tokenizer: Optional[GeneTokenizer] = None,
            
        """
        super().__init__()
        self.config = config
        self.model = torch.compile(model)
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore = ['model'])
        

    def _loss(self, batch: dict[str, Tensor], mode: Literal['train', 'valid']) -> Tensor:
        """loss

        Args:
            batch (Dict[str, Tensor]): generic batch, what your dataloader returns
            mode (Literal['train', 'valid']): who's calling

        Returns:
            Tensor: loss
        """
        loss = self.model.loss(batch)
        self.log(mode + "_loss", loss, prog_bar = True, sync_dist = True, on_step = True, on_epoch = True, batch_size = batch['x'].size(0))
        return loss
    


    def training_step(self, batch: Union[tuple[Tensor], dict[str, Tensor]]) -> Tensor:
        """training step

        Args:
            batch (Union[Tuple[Tensor], Dict[str, Tensor]]): generic batch, what your dataloader returns
            
        Returns:
            Tensor: loss
        """

        loss = self._loss(batch, 'train')
        return loss
    
    def validation_step(self, batch: Union[tuple[Tensor], dict[str, Tensor]]) -> None:
        """training step

        Args:
            batch (Union[Tuple[Tensor], Dict[str, Tensor]]): generic batch, what your dataloader returns
            batch_idx (int): batch index
            
        Returns:
            Tensor: loss
        """
        
        self._loss(batch, 'valid')
        
            

    def configure_optimizers(self) -> dict[str, Union[optim.Optimizer, CosineLRScheduler, optim.lr_scheduler.StepLR]]:
        """Put the optimizer in a dictionary with key 'optimizer' and the scheduler with key 'lr_scheduler'
        
        Returns:
            Dict[str, Union[optim.Optimizer, CosineLRScheduler, optim.lr_scheduler.StepLR]]: optimizer and scheduler"""

        out = dict()
        
        if self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), eps = 1e-8, betas = (0.9, 0.999), lr = self.config.learning_rate, weight_decay = 0.05)
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = self.config.learning_rate, momentum = 0.9, weight_decay = 0.001)
        else:
            raise NotImplementedError("This optimizer not implemented")
        out['optimizer'] = optimizer
        
        if self.config.scheduler == 'cosinelr':
            warmup_lr = 5e-7  
            warmup_epochs = 10
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, warmup_lr / self.config.learning_rate, 1, warmup_epochs)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 1, warmup_lr)
            out['lr_scheduler'] = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [warmup_scheduler, cosine_scheduler], milestones = [warmup_epochs])
            # scheduler = CosineLRScheduler(optimizer, t_initial = self.config.num_epochs - 20, lr_min = 5e-6, warmup_lr_init = 5e-7, warmup_t = 20, cycle_limit = 4, cycle_decay = 0.5, t_in_epochs = False, warmup_prefix = True)
        elif self.config.scheduler == 'steplr':
            out['lr_scheduler'] = optim.lr_scheduler.StepLR(out['optimizer'], step_size = self.config.step_size, gamma = 0.1)
        elif self.config.scheduler == 'none':
            pass
        else:
            raise NotImplementedError("This scheduler not implemented")
        
        return out