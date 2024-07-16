# real-wordpiece

**This repository provides a Python implementation of the `WordPiece` tokenizer, which is compatible with the 🤗
tokenizers.**

## Motivation

[🤗 tokenizers](https://github.com/huggingface/tokenizers) provides state-of-the-art text tokenizer implementations.
They are also used in the [🤗 transformers](https://github.com/huggingface/transformers), helping to convert text into
a format that might be fed into LLMs or embedding models. 🤗 tokenizers are broadly adopted in the NLP community, and
became the de-facto standard for tokenization, providing models such as:

- [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [WordPiece](https://huggingface.co/learn/nlp-course/en/chapter6/6)
- [Unigram](https://huggingface.co/learn/nlp-course/en/chapter6/7)

Surprisingly, the `WordPiece` tokenizer described in the brilliant [Hugging Face NLP
Course](https://huggingface.co/learn/nlp-course) is not the same as the one implemented in the 🤗 tokenizers library.
That fact is not well-known and vaguely documented. Instead of using the original `WordPiece` algorithm, it uses a
`##` prefix to indicate the continuation of the word and then uses the BPE algorithm to merge the tokens. This process
[is mentioned in the NLP Course](https://huggingface.co/learn/nlp-course/en/chapter6/6), but might be surprising for
those who haven't read it.

![HF tokenizers implementation of WordPiece](/docs/img/hf-wordpiece.png)

This library fills the gap by providing a Python implementation of the original `WordPiece` tokenizer, in a way that is
described in the course.

## Installation

`real-wordpiece` can be installed from PyPI using pip or the package manager of your choice:

```bash
pip install real-wordpiece
```

## Usage

Since the 🤗 tokenizers library is written in Rust, it is not possible to directly extend its interfaces with Python.
Thus, the `real-wordpiece` package provides a Python implementation of the `WordPiece` tokenizer, which can produce a
compatible model, but its interface is slightly different.

```python
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.tokenizers import Tokenizer

from real_wordpiece.trainer import RealWordPieceTrainer

# Create the Tokenizer, in the same way as you would with the 🤗 tokenizers
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = Lowercase()
tokenizer.pre_tokenizer = Whitespace()

# Define the training data
training_data = [
    "walker walked a long walk",
]

# Finally, train the tokenizer using the RealWordPieceTrainer
trainer = RealWordPieceTrainer(vocab_size=28, special_tokens=["[UNK]"])
trainer.train_tokenizer(training_data, tokenizer)

# The tokenizer.model will be now an instance of WordPiece trained above
print(tokenizer.encode("walker walked a long walk").tokens)
# Out: ['walk', '##er', 'walk', '##ed', 'a', 'long', 'walk']
```

In real-world applications the training corpus should be much larger, and the `vocab_size` should be set to a higher
value.

## WordPiece basics

WordPiece and Byte-Pair Encoding (BPE) are two of the most popular subword tokenization algorithms, and they have much
in common. Let's consider and example and assume we have just a single word in our corpus.

```python
word = "reappear"
```

The training process of the BPE algorithm starts with a vocabulary that contains all the characters.

```python
vocab = {"r", "e", "a", "p"}
```

The algorithm iteratively merges the most frequent pair of tokens in the vocabulary and adds it to the vocabulary,
until the vocabulary reaches the desired size. In the case of the word "reappear", the BPE algorithm would merge the
pair `("e", "a")` to create the token `"ea"`.

```python
vocab = {"r", "e", "a", "p", "ea"}
```

The process would continue until the vocabulary reaches the desired size or there are no more pairs to merge.

The WordPiece algorithm is similar to BPE, but it distinguishes first letters of words from the middle letters. The
middle letters are prefixed with `##`. The WordPiece algorithm starts with a vocabulary that contains all the characters
and also the middle letters.

```python
vocab = {"r", "e", "a", "p", "##r", "##e", "##a", "##p"}
```

WordPieces also uses a different heuristic to select the pair of tokens to merge. Instead of merging the most frequent
pair, WordPiece merges the pair that maximizes the score function that is defined as:

$$ score(u, v) = \frac{frequency(u, v)}{frequency(u) \dot frequency(v)} $$

Where $u$ and $v$ are tokens, $frequency(u, v)$ is the frequency of the pair in the corpus, and $frequency(u)$ and
$frequency(v)$ are the frequencies of the tokens $u$ and $v$ alone. BPE merges are a bit more intuitive, and both
algorithms may lead to different tokenization.

## Why does that matter?

Choosing the tokenization is another hyperparameter that can significantly affect the performance of the model. Large
Language Models' ability to solve different tasks might be limited by the tokenization algorithm used.

### References

The importance of the tokenization model is becoming more and more apparent. The following resources provide more
information on the topic:

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [WordPiece Pair Score Calculation and Reproducibility #1086](https://github.com/huggingface/tokenizers/issues/1086)
