
from torch import Tensor
from typing import Union

class SimpleTokenizer():
    """Simple tokenizer"""

    def __init__(self, tokens: list[str]):
        """Constructor

        Args:
            chars (list[str]): list of tokens in text
        """
        self.stoi = { ch : i for i, ch in enumerate(tokens) }
        self.itos = { i : ch for i, ch in enumerate(tokens) }
        self.tokens = tokens
        self.vocab_size = len(tokens)

    def tokenize(self, l: Union[list[str], str]) -> Union[list[int], int]:
        """Tokenizes into integers

        Args:
            l (Union[list[str], str]): tokens to tokenize

        Returns:
            Union[list[int], int]: tokenized tokens
        """
        return [self.stoi[c] for c in l] if isinstance(l, list) else self.stoi[l]
    
    def detokenize(self, l: list[int]) -> Union[list[str], str]:
        """Detokenizes into strings

        Args:
            l (list[int]): list of integers to detokenize

        Returns:
            Union[list[str], str]: detokenized tokens
        """
        return [self.itos[i] for i in l if i in self.itos] if isinstance(l, list) else self.itos[l]

    
    def detokenize_batch(self, batch: Tensor) -> list[list[str]]:
        """Detokenizes a batch of integers into strings

        Args:
            batch (torch.Tensor): tensor of shape (B x T)

        Returns:
            list[list[str]]: decoded text
        """
        return [self.detokenize(x.tolist()) for x in batch]
    
    def get_vocab_size(self) -> int:
        """Returns the vocab size

        Returns:
            int: vocab size
        """
        return self.vocab_size
    
    def get_tokens(self) -> list[str]:
        """Returns the tokens

        Returns:
            list[str]: tokens
        """
        return self.tokens