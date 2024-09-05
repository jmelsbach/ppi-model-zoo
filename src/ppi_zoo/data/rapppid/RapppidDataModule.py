from ppi_zoo.data.GoldStandardDataModule import GoldStandardDataModule
import numpy as np
import sentencepiece as sp

class RapppidDataModule(GoldStandardDataModule):
    def __init__(
        self,
        tokenizer_file: str,
        use_sentence_processor: bool = True,
        use_padding: bool = True,
        truncate_len: int = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_sentence_processor = use_sentence_processor
        self.use_padding = use_padding
        self.truncate_len = truncate_len
        sp.set_random_generator_seed(5353456) # todo: make this hyperparam
        self.tokenizer = sp.SentencePieceProcessor(model_file=tokenizer_file) 
        
    def transform_sequence(self, sequence: str) -> dict:
        tokens = sequence[:self.truncate_len]

        if self.use_sentence_processor:
            tokens = np.array(self.tokenizer.encode(tokens, enable_sampling=True, alpha=0.1, nbest_size=-1))
        if self.use_padding:
            pad_len = self.truncate_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), 'constant')

        return tokens