"""
This module is used for training a Sequence Tagger using WordEmbeddings and Flair Embeddings
"""
from typing import List

import click
import torch
from flair.data import Sentence
from flair.embeddings import FastTextEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import AdamW

from profile_extraction.ner_model.corpus import JsonlCorpus


class LowercaseFastTextEmbeddings(FastTextEmbeddings):  # pylint: disable=abstract-method
    """
    Class which lowercases FastTextEmbeddings for usage with lowercase only embeddings
    """

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        sentences_lower = [Sentence([t.text.lower() for t in sentence]) for sentence in sentences]
        super()._add_embeddings_internal(sentences_lower)

        for lower, sent in zip(sentences_lower, sentences):
            for idx, token in enumerate(lower):
                sent[idx].set_embedding(self.name, token.embedding)

        return sentences


@click.command()
@click.argument("data-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.argument("output-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.argument("embeddings-path", type=click.Path(dir_okay=False, file_okay=True), required=True)
def train_model(data_path: str, output_path: str, embeddings_path: str):
    """
    Trains a sequence tagger on the given data and saves it to output_path

    :param data_path: path to train, dev and test data
    :param output_path: path to save the model into
    """
    if not torch.cuda.is_available():
        print("GPU not available! Exiting...")
        return -1

    # 1. get the corpus
    corpus = JsonlCorpus(data_path)
    print(corpus)

    # 2. what label do we want to predict?
    label_type = "ner"

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)

    # 4. initialize embedding stack with Flair and GloVe
    embeddings = StackedEmbeddings(
        [
            LowercaseFastTextEmbeddings(embeddings_path),
            FlairEmbeddings("language_model/forward/best-lm.pt"),
            FlairEmbeddings("language_model/backward/best-lm.pt"),
        ]
    )

    # 5. initialize sequence tagger
    tagger = SequenceTagger(
        hidden_size=256, embeddings=embeddings, tag_dictionary=label_dict, tag_type=label_type, use_crf=True
    )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(
        output_path,
        learning_rate=1e-3,
        mini_batch_size=32,
        max_epochs=300,
        optimizer=AdamW,
        min_learning_rate=1e-6,
        mini_batch_chunk_size=16,
    )

    return 0
