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
