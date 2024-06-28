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
