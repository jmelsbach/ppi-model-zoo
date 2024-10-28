import numpy as np
import sentencepiece as sp

def tokenize(
        sequence: str,
        tokenizer_file: str,
        use_sentence_processor: bool = True,
        use_padding: bool = True,
        truncate_len: int = None,
        random_seed: int = 5353456
        ):
        
        sp.set_random_generator_seed(random_seed)
        tokenizer = sp.SentencePieceProcessor(model_file=tokenizer_file) 
        tokens = sequence[:truncate_len]

        if use_sentence_processor:
            tokens = np.array(tokenizer.encode(tokens, enable_sampling=False, alpha=0.1, nbest_size=-1))
        if use_padding:
            pad_len = truncate_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), 'constant')

        return tokens