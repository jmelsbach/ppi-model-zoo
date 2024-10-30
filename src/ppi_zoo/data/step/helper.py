from transformers import AutoTokenizer
import torch
import re


def tokenize(sequence: str, model: str, max_len: int = 1536):
    sequence = " ".join(sequence)
    sequence = re.sub(r"[UZOB]", "X", sequence)
    tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)
    tokens = tokenizer(
        sequence,
        max_length=max_len,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokens["input_ids"] = tokens["input_ids"].squeeze()
    tokens["attention_mask"] = tokens["attention_mask"].squeeze()
    tokens["token_type_ids"] = tokens["token_type_ids"].squeeze()

    return tokens
