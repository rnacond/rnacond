import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Optional
from src.utils.constants import BERT_MODEL_NAME, BERT_EMBEDDING_DIM
from transformers import DistilBertTokenizer
from torch import Tensor
import torch


class BERT(nn.Module):
    """Text encoder"""
    def __init__(self, pretrained: bool = True, trainable: bool = False) -> None:
        """Constructor

        Args:
            pretrained (bool, optional): Defaults to True.
            trainable (bool, optional): Defaults to True.
        """
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
        else:
            self.model = DistilBertModel(config = DistilBertConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.embedding_dim = BERT_EMBEDDING_DIM
        
    def get_token_embeddings(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Returns embeddings for all tokens

        Args:
            input_ids (torch.Tensor): tokens
            attention_mask (Optional[torch.Tensor]): attention mask

        Returns:
            torch.Tensor: embeddings
        """
        output = self.model(input_ids = input_ids, attention_mask = attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state

    def get_text_embedding(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Returns one embedding for the sentence

        Args:
            input_ids (torch.Tensor): tokens
            attention_mask (torch.Tensor): attention mask

        Returns:
            torch.Tensor: embedding
        """
        last_hidden_state = self.get_token_embeddings(input_ids, attention_mask)
        return last_hidden_state[:, self.target_token_idx, :]
    
    
    
def get_bert_embeddings(bert: BERT, bert_tokenizer: DistilBertTokenizer, text: list[str]) -> Tensor:
    """Get bert embeddings on CPU without batching, only if on short list of texts, e.g. some additional metadata terms 

    Args:
        bert (BERT): BERT model
        bert_tokenizer (DistilBertTokenizer): BERT tokenizer
        text (list[str]): list of strings to embed

    Returns:
        Tensor: embeddings of shape len(text) x bert.embedding_dim
    """
    
    tokens_ids = [bert_tokenizer(s, padding = 'longest', truncation = False) for s in text]
    embeddings = [bert.get_text_embedding(torch.tensor(x['input_ids']).long().reshape(1, -1), torch.tensor(x['attention_mask']).long().reshape(1, -1))[0] for x in tokens_ids]
    embeddings_tensor = torch.stack(embeddings, dim = 0)
    assert embeddings_tensor.size() == (len(text), bert.embedding_dim)
    return embeddings_tensor
