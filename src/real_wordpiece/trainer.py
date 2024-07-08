import logging
import sys
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordPiece

logger = logging.getLogger(__name__)


TokenPair = Tuple[str, str]


@dataclass
class Word:
    tokens: List[str] = field(default_factory=list)
    count: int = field(default=0)

    def add_token(self, token: str):
        self.tokens.append(token)

    def iter_tokens(self) -> Generator[str, None, None]:
        for token in self.tokens:
            yield token

    def iter_token_pairs(
        self, unique_only: bool = False
    ) -> Generator[TokenPair, None, None]:
        seen_pairs = set()
        for i in range(len(self.tokens) - 1):
            pair = (self.tokens[i], self.tokens[i + 1])
            if unique_only and pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            yield pair


class Tokenization:
    """
    Tokenization is a set of words that the set of texts was tokenized into. It keeps track of the frequency of each
    word and the tokens that compose it.
    """

    def __init__(self):
        self.words: List[Word] = []
        self.token_frequency: Dict[str, int] = Counter()
        self.pair_frequency: Dict[TokenPair, int] = Counter()
        self.scores: Dict[TokenPair, float] = OrderedDict()

    def add_word(self, word: Word):
        self.words.append(word)
        self.update_frequencies(word)

    def update_frequencies(self, word: Word):
        for token in word.iter_tokens():
            self.token_frequency[token] += word.count
        for pair in word.iter_token_pairs(unique_only=False):
            self.pair_frequency[pair] += word.count
        for pair in word.iter_token_pairs():
            self.scores[pair] = self.pair_frequency[pair] / (
                self.token_frequency[pair[0]] * self.token_frequency[pair[1]]
            )

    def iter_token_pairs(
        self, unique_only: bool = False
    ) -> Generator[TokenPair, None, None]:
        seen_pairs = set()
        for word in self.words:
            for pair in word.iter_token_pairs():
                if unique_only and pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                yield pair

    def get_symbol_frequency(self, symbol: str) -> int:
        return self.token_frequency[symbol]

    def get_pair_frequency(self, pair: TokenPair) -> int:
        return self.pair_frequency[pair]

    def merge_pair(self, pair: TokenPair, new_token: str):
        """
        Merge the pair of tokens into a new token.
        :param pair:
        :param new_token:
        :return:
        """
        recompute_score_pairs, remove_frequency_pairs = set(), {pair}
        for word in self.words:
            # Iterate from the end to the beginning to avoid index issues
            for index in range(len(word.tokens) - 2, -1, -1):
                current_pair = (word.tokens[index], word.tokens[index + 1])
                if current_pair == pair:
                    # Update the symbol frequency
                    self.token_frequency[new_token] += word.count

                    # Reduce the token frequency for the old symbols
                    self.token_frequency[word.tokens[index]] -= word.count
                    self.token_frequency[word.tokens[index + 1]] -= word.count

                    # Check the neighbors and update pair frequencies accordingly
                    if index - 1 >= 0:
                        left_pair = (word.tokens[index - 1], new_token)
                        self.pair_frequency[left_pair] += word.count
                        recompute_score_pairs.add(left_pair)
                        remove_frequency_pairs.add(
                            tuple(word.tokens[index - 1 : index + 1])
                        )
                    if index + 2 < len(word.tokens) - 1:
                        right_pair = (new_token, word.tokens[index + 2])
                        self.pair_frequency[right_pair] += word.count
                        recompute_score_pairs.add(right_pair)
                        remove_frequency_pairs.add(
                            tuple(word.tokens[index + 1 : index + 3])
                        )

                    # Replace the pair with the new token
                    word.tokens[index] = new_token
                    del word.tokens[index + 1]

        # Recalculate the scores for the newly created pairs
        for affected_pair in recompute_score_pairs:
            self.scores[affected_pair] = self.pair_frequency[affected_pair] / (
                self.token_frequency[affected_pair[0]]
                * self.token_frequency[affected_pair[1]]
            )

        # Remove the old pairs from the scores
        for affected_pair in remove_frequency_pairs:
            self.scores.pop(affected_pair, 0)  # noqa

        # Remove the old pairs from the frequencies
        for affected_pair in remove_frequency_pairs:
            self.pair_frequency.pop(affected_pair, 0)  # noqa


class RealWordPieceTrainer:
    """
    Trainer for the WordPiece algorithm using score-based selection for merge. The Hugging Face tokenizers
    implementation just adds the `##` prefix to all the middle letters of the words and then run the Byte-Pair Encoding
    training algorithm. That gives a good performance, but it is not the original WordPiece algorithm.

    RealWordPiece algorithm is a score-based selection for merge. Score for each pair of tokens is calculated as:

    score(u, v) = frequency(uv) / (frequency(u) * frequency(v))

    The pair of tokens with the maximum score is merged into a new token. This process is repeated until the vocabulary
    reaches the desired size.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        show_progress: bool = True,
        special_tokens: Optional[List[Union[str, AddedToken]]] = None,
        limit_alphabet: Optional[int] = None,
        initial_alphabet: Optional[Iterable[str]] = None,
        continuing_subword_prefix: str = "##",
        end_of_word_suffix: Optional[str] = None,
    ):
        """
        Trainer capable of training a WordPiece model using a score-based selection for merge.
        :param vocab_size: The size of the final vocabulary, including all tokens and alphabet.
        :param min_frequency: The minimum frequency a pair should have in order to be merged.
        :param show_progress: Whether to show progress bars while training.
        :param special_tokens: A list of special tokens the model should know of.
        :param limit_alphabet: The maximum different characters to keep in the alphabet.
        :param initial_alphabet:
            A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the
            strings contain more than one character, only the first one is kept.
        :param continuing_subword_prefix: A prefix to be used for every subword that is not a beginning-of-word.
        :param end_of_word_suffix: A suffix to be used for every subword that is a end-of-word.
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.show_progress = show_progress
        self.special_tokens = special_tokens or []
        self.limit_alphabet = limit_alphabet or sys.maxsize
        self.initial_alphabet = initial_alphabet or []
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix or ""

    def train_tokenizer(self, training_data: List[str], tokenizer: Tokenizer):
        """
        Train a tokenizer using the WordPiece algorithm. It modifies the tokenizer model in place.
        :param training_data:
        :param tokenizer:
        :return:
        """
        # Make sure the tokenizer model is a WordPiece instance
        assert isinstance(
            tokenizer.model, WordPiece
        ), "The tokenizer model must be a WordPiece instance."

        # Configure the underlying model
        tokenizer.model.continuing_subword_prefix = self.continuing_subword_prefix

        # Normalize the training data
        normalized_training_data = (
            [tokenizer.normalizer.normalize_str(text) for text in training_data]
            if tokenizer.normalizer
            else training_data
        )

        # Calculate the word counts
        word_counts = self.calculate_word_counts(normalized_training_data)

        # Initialize the vocabulary with the special tokens
        vocabulary: Dict[str, int] = OrderedDict()
        for token in self.special_tokens:
            vocabulary[token] = len(vocabulary)

        # Find the alphabet used in the normalized data
        alphabet = self.compute_alphabet(word_counts)
        for char in alphabet:
            vocabulary[char] = len(vocabulary)

        # Tokenize the words initially
        tokenization = self.init_tokenize_words(word_counts, vocabulary)

        while len(vocabulary) < self.vocab_size:
            # Filter only the scores that are above the minimum frequency
            filtered_scores = {
                pair: score
                for pair, score in tokenization.scores.items()
                if tokenization.get_pair_frequency(pair) >= self.min_frequency
            }

            # Break the process if there are no more pairs to merge
            if len(filtered_scores) == 0:
                logger.debug("No more pairs to merge")
                break

            # Find the pair with the maximum score, merge it and update the vocabulary
            max_pair, max_score = max(filtered_scores.items(), key=lambda x: x[1])
            new_token = max_pair[0] + max_pair[1].lstrip(self.continuing_subword_prefix)
            tokenization.merge_pair(max_pair, new_token)
            vocabulary[new_token] = len(vocabulary)
            logger.debug(f"Merged {max_pair} into {new_token} with score {max_score}")

        # Store the vocabulary in a temporary file and then load WordPiece model from it
        with tempfile.NamedTemporaryFile("w") as fp:
            fp.writelines([f"{line}\n" for line in vocabulary.keys()])
            fp.flush()

            # Load the model from the temporary file
            model = WordPiece.from_file(fp.name)

            # Reset the tokenizer model to the new one
            tokenizer.model = model

    def calculate_word_counts(self, texts: List[str]) -> Dict[str, int]:
        """
        Count the number of times each word appears in the provided texts.
        :param texts:
        :return:
        """
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        return word_counts

    def compute_alphabet(self, word_counts: Dict[str, int]) -> List[str]:
        """
        Determine the initial alphabet to use for the WordPiece algorithm.
        :param word_counts:
        :return:
        """
        # Compute the alphabet from seen words
        alphabet_counts = Counter()
        for word, count in word_counts.items():
            alphabet_counts.update(list(word) * count)

        # Also include anything from the provided initial alphabet
        for char in self.initial_alphabet:
            # Set to the integer max value
            alphabet_counts[char] = sys.maxsize

        # Remove the unwanted chars
        most_common = alphabet_counts.most_common(self.limit_alphabet)

        # Return the alphabet
        alphabet = [char for char, _ in most_common]
        return alphabet

    def init_tokenize_words(
        self, word_counts: Dict[str, int], vocabulary: Dict[str, int]
    ) -> Tokenization:
        """
        Tokenize the words using the initial vocabulary with the special tokens and the alphabet.
        :param word_counts:
        :param vocabulary:
        :return:
        """
        tokenization = Tokenization()
        for word, count in word_counts.items():
            tokenized_word = Word(count=count)
            for i, symbol in enumerate(word):
                is_first = i == 0
                is_last = i == len(word) - 1

                # Add the continuation prefix if needed
                if not is_first:
                    symbol = self.continuing_subword_prefix + symbol

                # Add the end of word suffix if needed
                if is_last:
                    symbol = symbol + self.end_of_word_suffix

                # Add the symbol to the vocabulary, if it doesn't exist
                if symbol not in vocabulary:
                    vocabulary[symbol] = len(vocabulary)

                # Add the symbol to the word sequence
                tokenized_word.add_token(symbol)

            # Store the tokenized word
            tokenization.add_word(tokenized_word)

        return tokenization
