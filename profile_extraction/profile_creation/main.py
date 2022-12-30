"""
This module provides commands to extract profiles from a JSON chat export
using a trained NER model
"""
from typing import TextIO

import click
from flair.models import RelationExtractor, SequenceTagger

from profile_extraction.profile_creation.extraction.profile_extractor import (
    extract_profiles,
)
from profile_extraction.profile_creation.reader import ChatReaderFactory, DataType


@click.command()
@click.option(
    "--model",
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to a SequenceTagger providing NER for the Profile Extraction",
)
@click.option(
    "--relation-model",
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to a RelationExtractor providing RC for the Profile Extraction",
)
@click.option("--chat", type=click.File(), required=True, help="File containing a JSON chat export")
@click.option("--output", type=click.File("w"), required=True, help="File to write the result to")
@click.option("--summary-only/--full-profiles", default=False, type=bool)
def profile_creation(model: str, relation_model: str, chat: TextIO, output: TextIO, summary_only: bool):
    """
    Generates Chat Profiles from a given chat export

    Args:
        model: Model to use for NamedEntityRecognition
        chat: Chat file to analyze
        output: File to save the generated profiles to
        summary_only: Whether the application should only generate a summary for each user
        relation_model: Model to use for RelationExtraction
    """
    reader = ChatReaderFactory.get_instance(DataType.JSON)
    sequence_tagger = SequenceTagger.load(model)
    relation_extractor = RelationExtractor.load(relation_model)

    profiles = extract_profiles(
        reader(chat),
        sequence_tagger=sequence_tagger,
        relation_extractor=relation_extractor,
        summary_only=summary_only,
    )
    print(profiles.json(indent=4), file=output)


if __name__ == "__main__":
    profile_creation()  # pylint: disable=no-value-for-parameter
