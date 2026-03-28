import torch
import torch.nn as nn   
import torch.nn.functional as F


class Residual(nn.Module):
    "Adds a residual connection"

    def __init__(self, fn: nn.Module) -> None:
        """ Constructor
        Args:
            fn (torch.nn.Module): torch module to add a skip connection to
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """ Forward function
        
        Args:
            x (torch.Tensor): input
            *args: arguments to self.fn
            **kwargs: keyword arguments to self.fn
            
        Returns:
            torch.Tensor: output
        """
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module):
    """RMS normalization"""

    def __init__(self, dim: int) -> None:
        """RMS normalization

        Args:
            dim (int): dimension
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input
        
        Returns:
            torch.Tensor: output
        """

        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreRMSNorm(nn.Module):
    """Adds RMS normalization before the given module"""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Constructor
        
        Args:
            dim (int) : dimension
            fn (torch.nn.Module) : mddule to apply an RMS norm before 
        """
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): input
            *args: arguments to self.fn
            **kwargs: keyword arguments to self.fn

        Returns:
            torch.Tensor: output
        """

        x = self.norm(x)
        return self.fn(x, *args, **kwargs)