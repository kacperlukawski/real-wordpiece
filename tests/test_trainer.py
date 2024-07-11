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

    trainer = RealWordPieceTrainer(vocab_size=29)
    trainer.train_tokenizer(training_data, tokenizer)

    assert len(tokenizer.get_vocab()) == 29
    assert tokenizer.encode("walker").tokens == ["wa", "##lke", "##r"]


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
