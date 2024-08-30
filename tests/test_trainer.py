import tempfile
from typing import List

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from real_wordpiece.trainer import RealWordPieceTrainer


@pytest.fixture
def training_data() -> List[str]:
    return [
        "walker walked a long walk",
        "a walker",
        "a walk",
    ]


def test_trainer_can_be_instantiated():
    trainer = RealWordPieceTrainer(vocab_size=100)
    assert trainer is not None


@pytest.mark.integration
def test_train_from_iterator_updates_wordpiece(training_data: List[str]):
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = RealWordPieceTrainer(vocab_size=27)
    trainer.train_tokenizer(training_data, tokenizer)

    assert len(tokenizer.get_vocab()) == 27
    assert tokenizer.encode("walker").tokens == ["walk", "##er"]


@pytest.mark.integration
def test_min_frequency_omits_rare_tokens(training_data: List[str]):
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = RealWordPieceTrainer(vocab_size=27, min_frequency=2)
    trainer.train_tokenizer(training_data, tokenizer)

    assert tokenizer.encode("long").tokens == ["l", "##o", "##n", "##g"]


@pytest.mark.integration
def test_tokenizer_save_and_load_keeps_the_vocabulary(training_data: List[str]):
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = RealWordPieceTrainer(vocab_size=29)
    trainer.train_tokenizer(training_data, tokenizer)
    vocabulary = tokenizer.get_vocab()

    with tempfile.TemporaryDirectory() as tmp_path:
        tokenizer.save(f"{tmp_path}/real_wordpiece.json", pretty=True)

        loaded_tokenizer = Tokenizer.from_file(f"{tmp_path}/real_wordpiece.json")
        loaded_vocabulary = loaded_tokenizer.get_vocab()
        assert loaded_vocabulary == vocabulary


@pytest.mark.integration
def test_word_count_is_correctly_calculated(training_data: List[str]):
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = RealWordPieceTrainer(vocab_size=100)
    word_counts = trainer.calculate_word_counts(tokenizer, training_data)
    assert len(word_counts) == 5
    assert word_counts["walker"] == 2
    assert word_counts["walked"] == 1
    assert word_counts["a"] == 3
    assert word_counts["long"] == 1
    assert word_counts["walk"] == 2


def test_alphabet_extracted_correctly(training_data: List[str]):
    word_counts = {
        "walker": 2,
        "walked": 1,
        "a": 3,
        "long": 1,
        "walk": 2,
    }

    trainer = RealWordPieceTrainer(vocab_size=100)
    alphabet = trainer.compute_alphabet(word_counts)
    target_letters = [
        "w",
        "a",
        "l",
        "k",
        "e",
        "r",
        "d",
        "o",
        "n",
        "g",
        "##a",
        "##l",
        "##k",
        "##e",
        "##r",
        "##d",
        "##o",
        "##n",
        "##g",
    ]

    assert set(alphabet) == set(target_letters)


def test_init_tokenization_calculates_values_properly():
    word_counts = {
        "walker": 1,
        "walked": 1,
        "a": 1,
        "long": 1,
        "walk": 1,
    }
    alphabet = [
        "w",
        "##a",
        "##l",
        "##k",
        "a",
        "##e",
        "##r",
        "l",
        "##d",
        "##o",
        "##n",
        "##g",
        "k",
        "e",
        "r",
        "d",
        "o",
        "n",
        "g",
    ]
    vocabulary = {letter: idx for idx, letter in enumerate(alphabet)}

    trainer = RealWordPieceTrainer(vocab_size=100)
    tokenization = trainer.init_tokenize_words(word_counts, vocabulary)

    # Make sure each individual token has a correct frequency
    assert tokenization.token_frequency["w"] == 3
    assert tokenization.token_frequency["##a"] == 3
    assert tokenization.token_frequency["##l"] == 3
    assert tokenization.token_frequency["##k"] == 3
    assert tokenization.token_frequency["##e"] == 2
    assert tokenization.token_frequency["##r"] == 1
    assert tokenization.token_frequency["##d"] == 1
    assert tokenization.token_frequency["a"] == 1
    assert tokenization.token_frequency["l"] == 1
    assert tokenization.token_frequency["##o"] == 1
    assert tokenization.token_frequency["##n"] == 1
    assert tokenization.token_frequency["##g"] == 1

    # Verify the pairs of tokens as well
    assert tokenization.pair_frequency[("w", "##a")] == 3
    assert tokenization.pair_frequency[("##a", "##l")] == 3
    assert tokenization.pair_frequency[("##l", "##k")] == 3
    assert tokenization.pair_frequency[("##k", "##e")] == 2
    assert tokenization.pair_frequency[("##e", "##r")] == 1
    assert tokenization.pair_frequency[("##e", "##d")] == 1
    assert tokenization.pair_frequency[("l", "##o")] == 1
    assert tokenization.pair_frequency[("##o", "##n")] == 1
    assert tokenization.pair_frequency[("##n", "##g")] == 1

    # Now check if scores are calculated correctly
    assert tokenization.scores[("w", "##a")] == 3 / (3 * 3)
    assert tokenization.scores[("##a", "##l")] == 3 / (3 * 3)
    assert tokenization.scores[("##l", "##k")] == 3 / (3 * 3)
    assert tokenization.scores[("##k", "##e")] == 2 / (3 * 2)
    assert tokenization.scores[("##e", "##r")] == 1 / (2 * 1)
    assert tokenization.scores[("##e", "##d")] == 1 / (2 * 1)
    assert tokenization.scores[("l", "##o")] == 1 / (1 * 1)
    assert tokenization.scores[("##o", "##n")] == 1 / (1 * 1)
    assert tokenization.scores[("##n", "##g")] == 1 / (1 * 1)


# TODO: test if training applies normalization and pre-tokenization
