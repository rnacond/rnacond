from typing import Any, Union, Optional, Literal
import torch
import random
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import anndata as ad




def rstrip_list(l: list[Any], to_strip: Any) -> list[Any]:
    """Right strips a list 

    Args:
        l (list[Any]): list
        to_strip (Any): token to strip

    Returns:
        list[Any]: stripped list
    """
    while len(l) > 0 and l[-1] == to_strip:
        l = l[:-1]
    return l

def grouped_shuffle(L: list[Any], shuffle: bool, order: Optional[list[Any]] = None) -> list[Any]:
    """Groups elements in a list then shuffles the groups, e.g [1,2,3,1,2,3] -> [2,2,3,3,1,1]

    Args:
        l (list[Any]): list
        shuffle (bool): shuffle or not
        order (Optional[list[Any]]): order if shuffle is False
        
    Returns:
        list[Any]: list
    """
    d = dict.fromkeys(L, 0) if order is None else dict.fromkeys(order, 0)
    for x in L:
        d[x] += 1
    keys = list(d.keys())
    if shuffle:
        random.shuffle(keys)
    R = []
    for key in keys:
        R.extend([key] * d[key])
    return R

def to_device(batch: Union[Tensor, tuple[Tensor], list[Tensor], dict[str, Tensor]], device: torch.device) -> Union[tuple[Tensor], list[Tensor], dict[str, Tensor]]:
    """Move to device
    Args:
        batch (Union[Tensor, tuple[Tensor], list[Tensor], dict[str, Tensor]]): batch
        device (torch.device): device
        
    Returns:
        Union[tuple[Tensor], list[Tensor], dict[str, Tensor]]: batch on device
    """
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items() if v is not None}
    if isinstance(batch, tuple):
        return [v.to(device) for v in batch if v is not None]
    if isinstance(batch, list):
        return [v.to(device) for v in batch if v is not None]
    if isinstance(batch, Tensor):
        return batch.to(device) if batch is not None else None
    
    


def sum_masks(mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Sums two masks

        Args:
            mask (Optional[Tensor], optional): Mask. Defaults to None.
            padding_mask (Optional[Tensor], optional): Padding mask. Defaults to None.

        Returns:
            Tensor: mask
        """
        if mask is None and padding_mask is None:
            return None
        if mask is not None and padding_mask is not None:
            if mask.size() != padding_mask.size():
                raise ValueError('Mask and padding mask should have the same size')
            return mask + padding_mask
        if mask is not None:
            return mask
        if padding_mask is not None:
            return padding_mask
        

def initialization(module: nn.Module, 
                   init_name: Literal['normal', 'xavier'] = 'xavier',
                   init_embeddings: bool = True
                   ) -> None:
    """Initializes weights

    Args:
        module (nn.Module): module
        init_name (Literal['normal', 'xavier']): initialization strategy
        init_embeddings (bool): initialize embeddings or not (if initialized to pretrained embeddings)
    """
    if isinstance(module, nn.Linear):
        if init_name == 'xavier':
            nn.init.xavier_normal_(module.weight)
        elif init_name == 'normal':
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        else :
            raise ValueError(f"Unknown initialization strategy '{init_name}'")
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding) and init_embeddings:
        nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.orthogonal_(module.weight, gain = 0.03)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)


exists = lambda x : x is not None # is not None function
        
            
def padded_stack(tensors: list[Tensor], 
                 side: Literal["left", "right"] = "right", 
                 mode: str = "constant", 
                 value: Union[int, float] = 0) -> tuple[Tensor, Union[Tensor, None]]:
    """Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (list[Tensor]): list of tensors to stack
        side (Literal['left', 'right']) : side on which to pad - "left" or "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        Tensor: stacked tensor
        Union[Tensor, None]: padding mask if padding was applied, None otherwise
    """
    sizes = [x.size(-1) for x in tensors]
    pad = sizes.count(sizes[0]) != len(sizes)
    if pad:
        size = max(sizes)
        padded, padding_mask = [], []
        for x in tensors:
            padding = make_padding(size - x.size(-1), side)
            padded.append(F.pad(x, padding, mode = mode, value = value) if size - x.size(-1) > 0 else x)
            padding_mask.append(torch.hstack([torch.zeros(x.size(-1)).bool(), torch.ones(size - x.size(-1)).bool()])) 
        return torch.vstack(padded), torch.vstack(padding_mask)
    else:
        return torch.vstack(tensors), None


def make_padding(pad: int, side: Literal["left", "right"] = "right") -> tuple[int]:
    """Returns proper input for torch.nn.functional.pad function

    Args:
        pad (int): number of elements to pad
        side (Literal['left', 'right']): side on which to pad - "left" or "right".

    Raises:
        ValueError: unknown side for padding

    Returns:
        tuple[int]: number of elements to pad to the left and to the right, in the last dimension of the tensor to pad
    """
    if side == "left":
        return (pad, 0)
    elif side == "right":
        return (0, pad)
    else:
        raise ValueError(f"side for padding '{side}' is unknown")
    

metadata_sentence = lambda metadata_name, metadata_value: metadata_name + ' is ' + str(metadata_value)  
    
def create_metadata_sentences(metadata_dict: dict[str, str], mask_prob: float = 0, delimiter: str = ',') -> str:
    """Sentence from metadata conditions

    Args:
        metadata_dict (dict[str, str]): dictionary of metadata field name to value
        mask_prob (float, optional): probability of not including one metadata value. Defaults to 0.
        delimiter (str): delimiter between metadata values

    Returns:
        str: text describing the metadata information
    """
    text = []
    for metadata_name, metadata_value in metadata_dict.items():
        if random.random() > mask_prob:
            text.append(metadata_sentence(metadata_name, metadata_value))
    text = (delimiter + ' ').join(text)
    return text


def prob_mask_like(shape: tuple[int], prob: float) -> Tensor:
    """Returns a boolean mask with a given probability of being True
    
    Args:
        shape (tuple[int]): shape of the mask
        prob (float): probability of being True
        
    Returns:
        Tensor: mask
    """

    if prob == 1:
        return torch.ones(shape, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, dtype = torch.bool)
    else:
        return torch.zeros(shape).float().uniform_(0, 1) < prob
    

identity = lambda x : x # identity function

none_if_not = lambda x, test : x if test else None # returns x if test else None


def to_array(matrix):
    if matrix is None:
        return None
    elif isinstance(matrix, ad.AnnData):
        matrix = matrix.X
    return matrix.toarray() if not isinstance(matrix, np.ndarray) else matrix

def extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    """Extracts (gathers) from tensor a at positions t

    Args:
        a (Tensor): tensor to extract from
        t (Tensor): positions to extract at
        x_shape (torch.Size): sample shape

    Returns:
        Tensor
    """

    b, *_ = t.shape
    out = a.gather(-1, t.type(torch.int64))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



to_neg_one_to_one = lambda x : x * 2 - 1 # From [0,1] to [-1,1]

to_zero_to_one = lambda x : (x + 1) * 0.5 # From [-1,1] to [0,1]


def unlog(samples):

    """Reverses log transform of the data

    Args:
        samples (numpy.array): data to unlog 
        
    Returns:
        samples (numpy.array): unloged data 
    """
    samples = np.round(np.exp(samples) - 1).astype(int)
    samples[samples < 0] = 0
    return samples

to_config = lambda config, config_class: config_class(**config) if isinstance(config, dict) else config


get_attributes = lambda obj, names: {name : getattr(obj, name, None) for name in names}


def extract_sublist(L: list[Any], K: list[Any]) -> list[Any]:
    """extracts a list from L with indices in K
    Args:
        L (list[Any]): list of data
        K (list[Any]): list of indices to extract from L
    Returns:
        list[Any]
    """
    return [x for (i,x) in enumerate(L) if i in K]