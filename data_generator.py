import numpy as np
import spicy as sp
import os

class Data_generator:

    def __init__(self, path: str):
        
        self.path = path

        with open(path) as f:
            data = f.read().lower()
        
        # all characters in data
        self.chars = list(set(data))
        self.char_to_index = {ch: i for (ch, i) in enumerate(self.chars)}
        self.index_to_char = {i: ch for (ch, i) in enumerate(self.chars)}

        self.vocab_size = len(self.chars)
        
        # read the examples
        with open(path) as f:
            examples = f.read().lower()

        self.examples = [x.lower().strip() for x in examples]

    def generate_example(self, index: int) -> np.array:

        example_chars = self.examples[index]
        example_char_index = [self.char_to_index[char] for char in example_chars]

        X = [self.char_to_index['\n']] + example_char_index
        Y = example_char_index + [self.char_to_index['\n']]

        return np.array(X), np.array(Y)
