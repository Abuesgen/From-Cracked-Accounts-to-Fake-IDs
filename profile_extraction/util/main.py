"""
This modules provides command line interfaces to some util functions.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO

import click

from profile_extraction.ner_model.corpus import JsonlDataset
from profile_extraction.util.utils import (
    filter_dataset,
    replace_nonprint,
    train_dev_test_split,
)


@click.command()
@click.option(
    "--dev-split",
    type=float,
    default=0.2,
    required=True,
    help="Percentage of the dataset which will be part of the development/validation set.",
)
@click.option(
    "--test-split",
    type=float,
    default=0.2,
    required=True,
    help="Percentage of the dataset which will be part of the test set.",
)
@click.option(
    "--random-state",
    type=int,
    default=None,
    help="Random state enabling reproducable shuffles.",
)
@click.argument("input-file", type=click.Path(dir_okay=False), required=True)
@click.argument("train-file", type=click.Path(dir_okay=False), required=True)
@click.argument("dev-file", type=click.Path(dir_okay=False), required=True)
@click.argument("test-file", type=click.Path(dir_okay=False), required=True)
def perform_split(
    dev_split: float, test_split: float, input_file, train_file, dev_file, test_file, random_state: Optional[int]
):
    """
    This method provides a command for splitting a datset file into train, dev and test files.
    Each line represents a data point.

    :param dev_split: Percentage of the dataset which will be part of the development/validation set.
    :param test_split: Percentage of the dataset which will be part of the test set.
    :param input_file: File containing all datapoints (one per line)
    :param train_file: File to write the train datapoints into
    :param dev_file: File to write the dev datapoints into
    :param test_file: File to write the test datapoints into
    :param random_state: Optional seed for reproducable shuffles
    """
    input_file = Path(input_file)
    train_file = Path(train_file)
    dev_file = Path(dev_file)
    test_file = Path(test_file)

    dataset = []
    with input_file.open(encoding="utf-8") as input_fp:
        dataset = input_fp.readlines()

    dataset = [json.loads(line) for line in dataset]
    dataset = filter_dataset(dataset)

    trainset, devset, testset = train_dev_test_split(dataset, dev_split, test_split, random_state=random_state)

    write_jsonl_file(trainset, train_file)
    write_jsonl_file(devset, dev_file)
    write_jsonl_file(testset, test_file)


def write_jsonl_file(data: List[Dict[str, Any]], file_path: Path):
    """
    Writes a list of Dictionaries to a jsonl file

    :param data: data to write into the jsonl file
    :param file_path: path to the file to write into
    """
    with file_path.open("w", encoding="utf-8") as file_pointer:
        for line in data:
            file_pointer.write(json.dumps(line))
            file_pointer.write("\n")


@click.command()
@click.option(
    "--dev-split",
    type=float,
    default=0.2,
    required=True,
    help="Percentage of the dataset which will be part of the development/validation set.",
)
@click.option(
    "--test-split",
    type=float,
    default=0.2,
    required=True,
    help="Percentage of the dataset which will be part of the test set.",
)
@click.option(
    "--random-state",
    type=int,
    default=None,
    help="Random state enabling reproducable shuffles.",
)
@click.argument("input-file", type=click.Path(dir_okay=False), required=True)
@click.argument("train-file", type=click.Path(dir_okay=False), required=True)
@click.argument("dev-file", type=click.Path(dir_okay=False), required=True)
@click.argument("test-file", type=click.Path(dir_okay=False), required=True)
def create_text_corpus(
    dev_split: float, test_split: float, input_file, train_file, dev_file, test_file, random_state: Optional[int]
):
    """
    Converts a jsonl corpus into a text corpus (it therefore removes all annotations).
    This helper function is used to pretrain FlairEmbeddings.

    :param dev_split: Percentage of the dataset which will be part of the development/validation set.
    :param test_split: Percentage of the dataset which will be part of the test set.
    :param random_state: Random state enabling reproducable shuffles.
    :param input_file: Jsonl file containing all datapoints
    :param train_file: File to write train datapoints into
    :param dev_file: File to write dev datapoints into
    :param test_file: FIle to write test datapoints into
    """
    input_file = Path(input_file)
    train_file = Path(train_file)
    dev_file = Path(dev_file)
    test_file = Path(test_file)

    dataset = []
    for sent in list(JsonlDataset(input_file)):
        dataset.append(" ".join([t.text.lower().strip() for t in sent]))

    trainset, devset, testset = train_dev_test_split(dataset, dev_split, test_split, random_state=random_state)

    write_text_file(trainset, train_file)
    write_text_file(devset, dev_file)
    write_text_file(testset, test_file)


def write_text_file(data: List[str], file_path: Path):
    """
    Writes multiple strings into a given file.
    It strips whitespaces and empty strings from the dataset.

    :param data: List of strings to write
    :param file_path: path to the output file
    """
    with file_path.open("w", encoding="utf-8") as file_pointer:
        for document in data:
            for line in document.split("\n"):
                if len(line.strip()) > 0:
                    file_pointer.write(line.strip())
                    file_pointer.write("\n")


@click.command()
@click.argument("annotation-path", type=click.Path(file_okay=True, dir_okay=False))
@click.argument("output-path", type=click.Path(file_okay=True, dir_okay=False))
def sanitize_data(annotation_path: str, output_path: str):
    """
    Sanatizes the text field inside an annotations json file
    """
    sanitized_data = []
    all_replaced: Set[str] = set()
    count = 0
    with Path(annotation_path).open("r", encoding="utf-8") as file_pointer:
        for line in file_pointer.readlines():
            data = json.loads(line)
            sanitized, replaced, num_replacements = replace_nonprint(data["text"])
            count = count + num_replacements
            data["text"] = sanitized
            sanitized_data.append(data)
            all_replaced.update(replaced)

    repl_list = list(all_replaced)
    print(f"Made {count} replacements of following chars:")
    print(sorted(repl_list))

    write_jsonl_file(sanitized_data, Path(output_path))


@click.command()
@click.argument("input-file", type=click.File(mode="r", encoding="utf-8"))
@click.argument("output-file", type=click.File(mode="w", encoding="utf-8"))
def prepare_relation_iaa(input_file: TextIO, output_file: TextIO):
    """
    This function  filters a jsonl testset, only preserving examples relevant for relation extraction.
    Further it removes relation and CRIT annotations to create a measureable IAA.
    """
    output_data = []
    for line in input_file.readlines():
        parsed = json.loads(line)

        skip = True  # Only allow lines containing CRIT and relation annotations
        current_line = {}
        for key, value in parsed.items():
            if key == "entities":  # Remove CRIT Entities and preserve others
                labels_data = []
                for label in value:
                    if label[2] != "CRIT":
                        labels_data.append(label)
                    else:
                        skip = False  # The example contains a CRIT, don't skip
                current_line[key] = labels_data
            elif key == "relations" and len(value) > 0:  # Example has relations, don't skip
                skip = False
            else:
                current_line[key] = value

        if not skip:
            output_data.append(current_line)

    output_file.writelines([json.dumps(dto) + "\n" for dto in output_data])


if __name__ == "__main__":
    sanitize_data()  # pylint: disable=no-value-for-parameter
