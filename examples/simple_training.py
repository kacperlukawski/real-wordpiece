from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace

from real_wordpiece.trainer import RealWordPieceTrainer

# Create the Tokenizer, in the same way as you would with the 🤗 tokenizers
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = Lowercase()
tokenizer.pre_tokenizer = Whitespace()

# Define the training data
training_data = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# Finally, train the tokenizer using the RealWordPieceTrainer
trainer = RealWordPieceTrainer(vocab_size=100, special_tokens=["[UNK]"])
trainer.train_tokenizer(training_data, tokenizer)

# The tokenizer.model will be now an instance of WordPiece trained above
print(tokenizer.encode("Hugging").tokens)
# Out: ['huggi', '##n', '##g']
print(tokenizer.encode("HOgging").tokens)
# Out: ['h', '##o', '##gg', '##i', '##n', '##g']
