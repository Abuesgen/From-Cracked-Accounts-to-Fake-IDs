"""
This module provides a command-line interface for relation extraction models
"""
import json
from pathlib import Path
from typing import Optional

import optuna
import torch
import typer
from flair.models import RelationExtractor

from profile_extraction.ner_model.corpus import JsonlCorpus, JsonlDataset
from profile_extraction.rel_model.train import (
    TrainConfig,
    optuna_objective,
    prepare_ner_model,
    train_rel_model,
)

app = typer.Typer()


def main():
    """
    Main entrypoint for the script (mandatory)
    """
    app()


@app.command()
def train(
    data_path: Path,
    output_path: Path,
    ner_model_path: Optional[Path] = None,
    learning_rate: float = 1e-5,
    mini_batch_size: int = 8,
    patience: int = 10,
    max_epochs: int = 800,
    model: str = "deepset/gbert-large",
    word_dropout: float = 0.1,
    locked_dropout: float = 0,
    dropout: float = 0.0,
):
    """
    Trains a transformer based SequenceTagger

    :param data_path: path to train, dev and test data
    :param output_path: path to save the trained model to
    """
    if not torch.cuda.is_available():
        print("GPU not available! Exiting...")
        return -1

    # 1. get the corpus
    corpus = JsonlCorpus(data_path, label_column_name="entities")
    print(corpus)

    ner_label_type = prepare_ner_model(corpus, ner_model_path)

    # 2. what label do we want to predict?
    label_type = "rel"
    train_config = TrainConfig(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        mini_batch_size=mini_batch_size,
        transformer_model=model,
        output_path=output_path,
        patience=patience,
        word_dropout=word_dropout,
        locked_dropout=locked_dropout,
        dropout=dropout,
    )

    train_rel_model(corpus, ner_label_type, label_type, training_config=train_config)

    return 0


@app.command()
def hyperparameter_search(
    data_path: Path,
    output_path: Path,
    n_trials: int = 3,
    n_epochs: int = 100,
    ner_model_path: Optional[Path] = None,
    study_name: str = "relation_hyperparameter_search",
    database: Optional[str] = None,
):
    """
    Performs a hyperparameter search using optuna
    Args:
        data_path: path to the JsonlCorpus
        output_path: path to persist the created trials and models
        n_trials: number of trials (tries) for this run
        n_epochs: number of epochs to train
        ner_model_path: path to an (optional) NER model. The predictions of the given model are used as GOLD
                        entities for training
        study_name: Name of the study. If it exists it is loaded otherwise it will be created
        database: optional database connection string for persisting the optuna runs

    Returns:
        int: 0 if the script finishes successfully -1 otherwise
    """
    if not torch.cuda.is_available():
        print("GPU not available! Exiting...")
        return -1

    # 1. get the corpus
    corpus = JsonlCorpus(data_path, label_column_name="entities")
    print(corpus)

    label_type: str = "rel"
    ner_label_type = prepare_ner_model(corpus, ner_model_path)

    def objective(trial):
        optuna_objective(corpus, output_path, ner_label_type, label_type, n_epochs, trial)

    study = optuna.create_study(direction="maximize", study_name=study_name, storage=database, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    result_text = [
        f"Best params: {study.best_params}",
        f"Best value: {study.best_value}",
        f"Best Trial: {study.best_trial}",
    ]

    for text in result_text:
        typer.echo(text)

    return 0


@app.command()
def evaluate(
    data_path: Path,
    model_path: Path,
    output_file: Path,
    ner_model_path: Optional[Path] = None,
):
    """
    Runs an evaluation on the given dataset
    """
    model = RelationExtractor.load(model_path)
    print(model)

    label_type: str = "rel"
    dataset = JsonlDataset(data_path)
    ner_label_type = prepare_ner_model(dataset, ner_model_path)

    model.entity_label_type = ner_label_type
    result = model.evaluate(data_points=dataset.sentences, gold_label_type=label_type)
    with output_file.open("w") as output_fp:
        json.dump(result.classification_report, output_fp)


if __name__ == "__main__":
    main()
