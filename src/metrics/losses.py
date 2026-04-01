import torch
import torch.nn.functional as F
from typing import Optional

def masked_cross_entropy(scores: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked cross entropy

        Args:
            scores (torch.Tensor): scores
            targets (torch.Tensor): targets
            mask (Optional[torch.Tensor], optional): mask. Defaults to None.
        
        Returns:
            torch.Tensor: loss
        """
        
        B, T, C = scores.shape
        scores = scores.view(B * T, C)
        targets = targets.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            scores = scores[mask, :]
            targets = targets[mask]
        loss = F.cross_entropy(scores, targets)
        return loss