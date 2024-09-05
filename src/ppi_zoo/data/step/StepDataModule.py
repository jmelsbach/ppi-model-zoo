from transformers import AutoTokenizer
import torch
import re
from ppi_zoo.data.GoldStandardDataModule import GoldStandardDataModule

class StepDataModule(GoldStandardDataModule):
    
    def __init__(
        self,
        tokenizer: str = None,
        max_len: int = 1536,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, do_lower_case=False
        )

    def transform_sequence(self, sequence: str) -> torch.Tensor:
        sequence = " ".join(sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        tokens = self.tokenizer(
            sequence,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        tokens['input_ids'] = tokens['input_ids'].squeeze()
        tokens['attention_mask'] = tokens['attention_mask'].squeeze()
        tokens['token_type_ids'] = tokens['token_type_ids'].squeeze()

        return tokens