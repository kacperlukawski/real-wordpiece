from pathlib import Path

from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer

from real_wordpiece.trainer import RealWordPieceTrainer

CURRENT_DIR = Path(__file__).parent

dataset = load_dataset("sentence-transformers/natural-questions", split="train")
training_data = dataset["query"] + dataset["answer"]

# 🤗 tokenizers implementation of the WordPiece algorithm
hf_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
hf_tokenizer.normalizer = BertNormalizer()
hf_tokenizer.pre_tokenizer = BertPreTokenizer()
hf_trainer = WordPieceTrainer(vocab_size=30000, special_tokens=["[UNK]"])
hf_tokenizer.train_from_iterator(training_data, hf_trainer)
hf_tokenizer.save(str(CURRENT_DIR / "tokenizers" / "hf_wordpiece.json"), pretty=True)
print("Finished training HF WordPiece")

# real-wordpiece implementation of the WordPiece algorithm
real_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
real_tokenizer.normalizer = BertNormalizer()
real_tokenizer.pre_tokenizer = BertPreTokenizer()
real_trainer = RealWordPieceTrainer(
    vocab_size=30000, special_tokens=["[UNK]"], min_frequency=10
)
real_trainer.train_tokenizer(training_data, real_tokenizer)
hf_tokenizer.save(str(CURRENT_DIR / "tokenizers" / "real_wordpiece.json"), pretty=True)
print("Finished training real WordPiece")

# Compare the tokenization results
hf_vocab = set(hf_tokenizer.get_vocab().keys())
real_vocab = set(real_tokenizer.get_vocab().keys())

# Display the differences in the vocabularies and their sizes
hf_but_not_real = hf_vocab - real_vocab
real_but_not_hf = real_vocab - hf_vocab

print(f"Tokens in HF WordPiece but not in Real WordPiece ({len(hf_but_not_real)}):")
print(hf_but_not_real)

print(f"Tokens in Real WordPiece but not in HF WordPiece ({len(real_but_not_hf)}):")
print(real_vocab - hf_vocab)
