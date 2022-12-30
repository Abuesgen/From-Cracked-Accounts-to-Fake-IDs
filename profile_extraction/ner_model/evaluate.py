"""
This module provides functions and commands for evaluating models
"""
import json

import click
from flair.models import SequenceTagger

from profile_extraction.ner_model.corpus import JsonlDataset


@click.command()
@click.option("-m", "--model-path", type=click.Path(dir_okay=False), required=True, help="Path to the model file")
@click.option(
    "-d", "--data-path", type=click.Path(dir_okay=False, file_okay=True), required=True, help="Path to the test data"
)
@click.argument("output_file", type=click.File(mode="w"))
def evaluate(model_path: str, data_path: str, output_file):
    """
    Evaluates a given model and writes a classification report to output_file

    :param model_path: Path to the model
    :param data_path: Path to test data with gold labels
    :param output_file: File handle to write the json classification report to
    """
    model = SequenceTagger.load(model_path)
    result = model.evaluate(data_points=JsonlDataset(data_path).sentences, gold_label_type="ner")
    json.dump(result.classification_report, output_file)
