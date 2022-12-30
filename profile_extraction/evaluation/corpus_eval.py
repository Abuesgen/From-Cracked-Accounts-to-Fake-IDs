"""
This module provides functions for corpus metrics
"""

import click
from flair.data import FlairDataset

from profile_extraction.ner_model.corpus import JsonlCorpus


@click.command()
@click.option("--corpus", type=click.Path(dir_okay=True, file_okay=False, exists=True), required=True)
def list_token_sizes(corpus: str):
    """
    command line tool to list corpus sizes in tokens.
    :param corpus: corpus to analyze
    """
    corpus = JsonlCorpus(corpus)

    print_stat("Train", corpus.train)
    print_stat("Dev", corpus.dev)
    print_stat("Test", corpus.test)


def print_stat(title: str, dataset: FlairDataset):
    """
    Prints a formatted string of dataset statistics
    :param title:  title to prepend to string
    :param dataset: dataset to print stats for
    """
    sentences, tokens, annotated = compute_sizes(dataset)
    print(f"{title}: Documents: {sentences}, Tokens: {tokens}, Annotated: {annotated}")


def compute_sizes(dataset: FlairDataset):
    """
    Computes length, token count and annotated token count for a given dataset
    :param dataset: dataset to compute metrics for
    """
    token_count = 0
    annotated_count = 0

    for sentence in dataset:
        token_count += len(sentence)
        for token in sentence:
            if token.get_tag("ner").value not in ["", "O"]:
                annotated_count += 1

    return len(dataset), token_count, annotated_count


if __name__ == "__main__":
    list_token_sizes()  # pylint: disable=no-value-for-parameter
