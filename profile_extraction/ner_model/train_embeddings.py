"""
This module provides functions and commands for training FlairEmbeddings
"""
import multiprocessing
from typing import List, Optional

import click
import gensim
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from gensim.models.fasttext import save_facebook_model
from torch.optim import AdamW

from profile_extraction.ner_model.corpus import JsonlDataset


@click.command()
@click.option("--forward/--backward", default=True)
@click.argument("data-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.argument("output-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
def train_language_model(data_path: str, output_path: str, forward: bool):
    """
    Trains a language model with a given dictionary

    :param data_path: path to train, dev and testdata
    :param output_path: path to save the trained model to
    :param forward: whether to train a forward or backward language model
    """

    # instantiate an existing LM, such as one from the FlairEmbeddings
    language_model = FlairEmbeddings(f'de-{"forward" if forward else "backward" }').lm

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(data_path, dictionary, is_forward_lm, character_level=True)

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus, optimizer=AdamW)

    trainer.train(
        output_path, sequence_length=250, mini_batch_size=100, max_epochs=100, patience=10, learning_rate=1e-3
    )


@click.command()
@click.argument("fasttext-path", type=click.Path(dir_okay=False, file_okay=True), required=True)
@click.argument("data-path", type=click.Path(dir_okay=False, file_okay=True), required=True)
@click.argument("output-path", type=click.Path(dir_okay=False, file_okay=True), required=True)
@click.option("--epochs", type=click.IntRange(min=1), default=None)
def train_fasttext_model(fasttext_path: str, data_path: str, output_path: str, epochs: Optional[int] = None):
    """
    Trains a language model with a given dictionary

    :param fasttext_path: Path to existing fasttext .bin file
    :param data_path: path to train, dev and testdata
    :param output_path: path to save the trained model to
    :param epochs: Number of epochs to train. Uses model Epochs by default
    """
    model = gensim.models.fasttext.load_facebook_model(fasttext_path)

    if epochs is not None:
        model.epochs = epochs

    model.workers = multiprocessing.cpu_count()

    # get your corpus, process forward and at the character level
    dataset = JsonlDataset(path_to_json_file=data_path)

    new_sentences: List[List[str]] = []
    for sent in list(dataset):
        new_sentences.append([t.text.lower().strip() for t in sent])

    model.build_vocab(new_sentences, update=True)
    print(f"Training {model.epochs} epochs for fine-tune.")
    model.train(new_sentences, total_examples=len(new_sentences), epochs=model.epochs)

    save_facebook_model(model, output_path)
