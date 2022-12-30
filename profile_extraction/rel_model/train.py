"""
This module provides a SequenceTagger training using transformer instead of word vectors
"""
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
from flair.data import FlairDataset
from flair.embeddings import TransformerWordEmbeddings
from flair.models import RelationExtractor, SequenceTagger
from flair.trainers import ModelTrainer
from optuna import Trial
from pydantic import BaseModel

from profile_extraction.ner_model.corpus import JsonlCorpus


def optuna_objective(
    corpus: JsonlCorpus,
    output_path: Union[str, Path],
    ner_label_type: str,
    label_type: str,
    n_epochs: int,
    trial: Trial,
) -> float:
    """
    Optuna objective for Hyperparameter search
    Args:
        corpus: JsonlCorpus containing train, dev and test dataset
        output_path: Path where the different trials are persisted (
        ner_label_type: name of the ner label namespace to use
        label_type: name of the relation label namespace to use
        n_epochs: number of epochs to train
        trial: optuna trial object which proposes the hyperparameters

    Returns:
        float: best model f1 score achieved by the executed training
    """
    corpus_copy = deepcopy(corpus)
    current_output_path = Path(output_path) / f"trial_{trial.number:04d}"

    optimizer_str = trial.suggest_categorical("optimizer", ["ADAM", "SGD"])
    if optimizer_str == "ADAM":
        optimizer = torch.optim.AdamW
    elif optimizer_str == "SGD":
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f"The given optimizer {optimizer_str} is not configured.")

    train_config = TrainConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True),
        max_epochs=n_epochs,
        mini_batch_size=16,
        transformer_model=trial.suggest_categorical(
            "transformer_embeddings", ["deepset/gbert-large", "Twitter/twhin-bert-large"]
        ),
        optimizer=optimizer,
        word_dropout=trial.suggest_float("word_dropout", 0.0, 0.2),
        locked_dropout=trial.suggest_float("locked_dropout", 0, 0.2),
        dropout=trial.suggest_float("dropout", 0.0, 0.2),
        output_path=current_output_path,
        patience=4,
    )

    return train_rel_model(corpus_copy, ner_label_type, label_type, training_config=train_config).test_score


def train_rel_model(
    corpus: JsonlCorpus, ner_label_type: str, label_type: str, training_config: "TrainConfig"
) -> "TrainingResult":
    """
    Trains a new relation extraction (classification) model on the given corpus
    Args:
        corpus: corpus containing train, dev and test dataset
        ner_label_type: label namespace containing ner labels
        label_type: label namespace containing relation labels
        training_config: Training parameters

    Returns:
        TrainingResult: Training metrics and loss history
    """
    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)
    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(
        model=training_config.transformer_model,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
    )
    classifier = RelationExtractor(
        embeddings,
        label_type=label_type,
        entity_label_type=ner_label_type,
        label_dictionary=label_dict,
        entity_pair_filters=[("PROD", "MONEY"), ("MONEY", "CRIT")],
        word_dropout_value=training_config.word_dropout,
        dropout_value=training_config.dropout,
        locked_dropout_value=training_config.locked_dropout,
    )
    trainer = ModelTrainer(classifier, corpus)
    return TrainingResult(
        **trainer.fine_tune(
            training_config.output_path,
            learning_rate=training_config.learning_rate,
            mini_batch_size=training_config.mini_batch_size,
            mini_batch_chunk_size=training_config.mini_batch_chunk_size,
            use_final_model_for_eval=False,
            patience=training_config.patience,
            max_epochs=training_config.max_epochs,
            main_evaluation_metric=("micro avg", "f1-score"),
            optimizer=training_config.optimizer,
            min_learning_rate=1e-7,
        )
    )


class TrainConfig(BaseModel):
    """
    This DTO aggregates all parameters for a model training
    """

    output_path: Union[Path, str]
    learning_rate: float = 3.18e-5
    max_epochs: int = 800
    transformer_model: str = "deepset/gbert-large"
    patience: int = 4
    optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW
    word_dropout: float = 0.1431
    locked_dropout: float = 0.0713
    dropout: float = 0.0253
    mini_batch_size: int = 16
    mini_batch_chunk_size = 8


class TrainingResult(BaseModel):
    """
    This DTO represents the dictionary returned by flairs model trainer
    """

    test_score: float
    dev_score_history: List[float]
    train_loss_history: List[float]
    dev_loss_history: List[float]


def prepare_ner_model(corpus: Union[JsonlCorpus, FlairDataset], ner_model_path: Optional[Union[Path, str]]) -> str:
    """
    This helper function loads a SequenceTagger (if path is set) and tags the corpus using the loaded Tagger.
    This can be used to train the relation classification model on predictions.

    Args:
        ner_model_path: Path or name of the NER model to load
        corpus (JsonlCorpus): Corpus to which the predicted NEs are appended

    Returns:
        str: name of the ner label type in the corpus. If ner_Model_path is None it is 'ner' else it is 'pred_ner'

    """
    ner_label_type = "ner"
    if ner_model_path:
        ner_label_type = "pred_ner"
        ner_model = SequenceTagger.load(ner_model_path)
        ner_model.predict(
            list(corpus.get_all_sentences() if isinstance(corpus, JsonlCorpus) else corpus), label_name=ner_label_type
        )
    return ner_label_type
