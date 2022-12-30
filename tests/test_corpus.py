"""
This module contains tests for JsonlCorpora
"""
import pytest
from flair.data import Sentence

from profile_extraction.ner_model.corpus import (
    JsonlCorpus,
    JsonlDataset,
    MultiFileJsonlCorpus,
)


class TestJsonlDataset:
    """
    Test the JsonlDataset class
    """

    def test_reading_dataset_with_one_entry_should_be_successful(self):
        """
        Tests reading a JsonlDataset containing a single entry
        """
        dataset = JsonlDataset("tests/resources/test_dataset.jsonl", label_column_name="labels")

        assert len(dataset.sentences) == 1
        assert dataset.sentences[0].to_tagged_string() == "Die Uhren <B-PROD> ?"

    @pytest.mark.parametrize(
        "input_text,labels,expected",
        [
            ("Die Uhren", [[4, 9, "PROD"]], "Die Uhren <B-PROD>"),
            ("Die Uhren?", [[4, 9, "PROD"]], "Die Uhren <B-PROD> ?"),
            ("Die Uhren?", [[4, 10, "PROD"]], "Die Uhren <B-PROD> ? <I-PROD>"),
        ],
    )
    def test_extract_single_lable_should_be_successful(self, input_text, labels, expected):
        """
        Tests whether labels are correctly applied to sentences
        """
        sentence = Sentence(input_text)
        JsonlDataset._add_labels_to_sentence(input_text, sentence, labels)

        assert sentence.to_tagged_string() == expected


class TestJsonlCorpus:
    """
    Tests the JsonlCorpus class
    """

    def test_simple_folder_corpus_should_load(self):
        corpus = JsonlCorpus("tests/resources/test_corpus/")
        assert len(corpus.get_all_sentences()) == 30


class TestMultiFileJsonlCorpus:
    """
    Tests the MultiFileJsonlCorpus
    """

    @pytest.mark.parametrize(
        "train_files,dev_files,test_files,expected_size",
        [
            (
                ["tests/resources/test_corpus/train.jsonl"],
                ["tests/resources/test_corpus/dev.jsonl"],
                ["tests/resources/test_corpus/test.jsonl"],
                30,
            ),
            (
                ["tests/resources/test_corpus/train.jsonl"],
                [],
                ["tests/resources/test_corpus/test.jsonl"],
                20,
            ),
            (
                ["tests/resources/test_corpus/train.jsonl"],
                [],
                None,
                10,
            ),
            (
                None,
                ["tests/resources/test_corpus/dev.jsonl"],
                None,
                10,
            ),
            (
                ["tests/resources/test_corpus/train.jsonl", "tests/resources/test_corpus/dev.jsonl"],
                ["tests/resources/test_corpus/dev.jsonl"],
                ["tests/resources/test_corpus/test.jsonl"],
                40,
            ),
        ],
    )
    def test_corpus_with_single_files_should_load(self, train_files, dev_files, test_files, expected_size):
        corpus = MultiFileJsonlCorpus(train_files, dev_files, test_files)
        assert len(corpus.get_all_sentences()) == expected_size

    def test_empty_corpus_should_raise_error(self):
        with pytest.raises(RuntimeError) as err:
            MultiFileJsonlCorpus(None, None, None)

        assert str(err.value) == "No data provided when initializing corpus object."
