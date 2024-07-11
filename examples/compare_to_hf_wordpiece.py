import logging
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFKD, Lowercase, Sequence, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.pre_tokenizers import Sequence as PreTokenizationSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer

from real_wordpiece.trainer import RealWordPieceTrainer

CURRENT_DIR = Path(__file__).parent
VOCAB_SIZE = 30000
MIN_FREQUENCY = 5

dataset = load_dataset("TopicNavi/Wikipedia-example-data", split="train")
training_data = dataset["text"]

# Enable logging to console (level INFO) and to a file at the same time (level DEBUG)
# Filename for the log should contain the launch datetime of the script to allow multiple
# run logs in the same directory.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(
    str(CURRENT_DIR / "logs" / f"{datetime.now()}-compare_to_hf_wordpiece.log"),
    mode="w",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configure pre-tokenization and normalization
normalizer = Sequence([NFKD(), Lowercase(), StripAccents(), Strip()])
pre_tokenizer = PreTokenizationSequence([Whitespace(), Punctuation()])

# 🤗 tokenizers implementation of the WordPiece algorithm
hf_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
hf_tokenizer.normalizer = normalizer
hf_tokenizer.pre_tokenizer = pre_tokenizer
hf_trainer = WordPieceTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]"],
    min_frequency=MIN_FREQUENCY,
)
hf_tokenizer.train_from_iterator(training_data, hf_trainer)
hf_tokenizer.save(str(CURRENT_DIR / "tokenizers" / "hf_wordpiece.json"), pretty=True)
print("Finished training HF WordPiece")

# real-wordpiece implementation of the WordPiece algorithm
real_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
real_tokenizer.normalizer = normalizer
real_tokenizer.pre_tokenizer = pre_tokenizer
real_trainer = RealWordPieceTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]"],
    min_frequency=MIN_FREQUENCY,
)
real_trainer.train_tokenizer(training_data, real_tokenizer)
real_tokenizer.save(
    str(CURRENT_DIR / "tokenizers" / "real_wordpiece.json"), pretty=True
)
print("Finished training real WordPiece")

# Compare the tokenization results
hf_vocab = set(hf_tokenizer.get_vocab().keys())
real_vocab = set(real_tokenizer.get_vocab().keys())

# Display the differences in the vocabularies
hf_but_not_real = hf_vocab - real_vocab
real_but_not_hf = real_vocab - hf_vocab

print(f"Tokens in HF WordPiece but not in Real WordPiece ({len(hf_but_not_real)}):")
print(hf_but_not_real)

print(f"Tokens in Real WordPiece but not in HF WordPiece ({len(real_but_not_hf)}):")
print(real_vocab - hf_vocab)
