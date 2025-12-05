# FT Tokenize

I created this small C++17 tokenizer library as a personal project to better understand how tokenization works.
<br>It supports both word-level and BPE (Byte Pair Encoding) tokenization and provides a simple Python API via pybind11.
<br>The library is also published on **PyPI**, so it can be easily installed with pip to train, save, load, encode, and decode text.
<br>A [demo notebook](./ft-tokenize.ipynb) demonstrating how to use the library is available in the root of this repository and is linked with a [Kaggle notebook](https://www.kaggle.com/code/tommasofacchin/ft-tokenize).

---

## Features

- Word-level tokenizer  
- BPE tokenizer with merge-rule learning  
- Training from text files  
- Vocabulary export / import  
- Token ↔ ID conversions  
- Thread-safe operations using `std::mutex`  
- Python bindings via `pybind11`

---

## Requirements
 
- Python 3.8+

---
 
## Installation

You can install the package directly with:

```sh
pip install ft_tokenize
```

---

## Python Usage Example

```python
from ft_tokenize import TokenizerModel

# Create a tokenizer
tok = TokenizerModel()

# Train from a text file
tok.train_from_textfile(
    "data.txt", 
    vocab_size=10000, 
    user_defined_symbols=["<sos>", "<eos>"], 
    mode="BPE"
)


# Save / load model
tok.save_model("vocab.txt")
tok.load_model("vocab.txt")

# Encode / decode some sample text
ids = tok.encode_as_ids("My name is Tommaso")
tokens = tok.encode_as_tokens("My name is Tommaso")

# Print the token IDs to see how the text is split internally
print(ids)  # Example output: [5, 8, 9, 12]

# Print the actual token strings
print(tokens)  # Example output: ['My', 'name', 'is', 'Tommaso']

# Decode back from IDs and tokens to check correctness
print(tok.decode_ids(ids))        # Should return "My name is Tommaso"
print(tok.decode_tokens(tokens))  # Should return "My name is Tommaso"

# Check vocabulary details
print("Vocab size:", tok.get_token_size())        # How many tokens are in the vocabulary
print("Token for ID 10:", tok.id_to_token(10))    # Look up a token by its ID
```

---

## Project Structure

```
src/
  ft_tokenizer.cpp        # pybind11 module
  tokenizer_model.hpp     # Tokenizer class definition
  tokenizer_model.cpp     # Implementation
```

---

## How It Works

### WORD mode

* Splits text by whitespace
* Builds a frequency-sorted vocabulary
* Maps unknown words to `<unk>`

### BPE mode

* Splits words into characters
* Iteratively merges the most frequent symbol pairs
* Stores merge rules inside the vocabulary
* Encodes text greedily (longest match)



The `TokenizerModel` class provides the following methods:

### Training

* `train_from_textfile(input_file, vocab_size=10000, user_defined_symbols=[], mode="WORD")`
  Train the tokenizer from a text file.

  * `input_file`: path to a text file for training
  * `vocab_size`: maximum number of tokens in the vocabulary
  * `user_defined_symbols`: list of extra tokens to include
  * `mode`: `"WORD"` or `"BPE"`

* `train_word_level(input_file, vocab_size, user_defined_symbols)`
  Train a word-level tokenizer. Usually called internally.

* `train_bpe(input_file, vocab_size, user_defined_symbols)`
  Train a BPE tokenizer. Usually called internally.


### Saving and Loading

* `save_model(model_path)`
  Save the current vocabulary to a file.

* `load_model(model_path)`
  Load a vocabulary from a file.


### Encoding

* `encode_as_ids(text)` → List of integers
  Convert a string into token IDs.

* `encode_as_tokens(text)` → List of strings
  Convert a string into token strings.


### Decoding

* `decode_ids(ids)` → String
  Convert a list of token IDs back into a string.

* `decode_tokens(tokens)` → String
  Convert a list of token strings back into a string.


### Utility

* `token_to_id(token)` → Integer
  Get the ID of a token. Returns the `<unk>` ID if the token is not found.

* `id_to_token(id)` → String
  Get the token corresponding to an ID. Returns `<unk>` if the ID is invalid.

* `get_token_size()` → Integer
  Returns the size of the vocabulary.

* `get_vocab()` → List of strings
  Returns the full vocabulary.

