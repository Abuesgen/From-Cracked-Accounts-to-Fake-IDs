"""
This module provides a SequenceTagger training using transformer instead of word vectors
"""
import click
import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from profile_extraction.ner_model.corpus import JsonlCorpus


@click.command()
@click.argument("data-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.argument("output-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
def train_model(data_path: str, output_path: str):
    """
    Trains a transformer based SequenceTagger

    :param data_path: path to train, dev and test data
    :param output_path: path to save the trained model to
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

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(
        model="deepset/gbert-large",
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
    )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type="ner",
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. run fine-tuning
    trainer.fine_tune(
        output_path,
        learning_rate=1e-5,
        mini_batch_size=4,
        use_final_model_for_eval=False,
        max_epochs=800,
        main_evaluation_metric=("micro avg", "f1-score"),
    )

    return 0
