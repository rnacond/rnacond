from src.utils.constants import BERT_MODEL_NAME
from transformers import DistilBertTokenizer



def get_bert_tokenizer(bert_model_name: str = BERT_MODEL_NAME) -> DistilBertTokenizer:
    """Returns a DistilBertTokenizer

    Args:
        bert_model_name (str, optional): bert model name. Defaults to BERT_MODEL_NAME.

    Returns:
        DistilBertTokenizer: tokenizer
    """
    return DistilBertTokenizer.from_pretrained(bert_model_name, clean_up_tokenization_spaces = True) 