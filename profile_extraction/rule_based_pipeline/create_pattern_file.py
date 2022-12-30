"""
Creates a file of string patterns from train data
"""
# pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name,missing-function-docstring,too-many-locals
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import click
import srsly
from flair.tokenization import SegtokTokenizer


def create_phrase_pattern(phrase):
    return [{"LOWER": p} if i < 2 else {"LOWER": p, "OP": "?"} for i, p in enumerate(phrase)]


def create_patterns(train_sample_it: Iterable[Dict]):
    tokenizer = SegtokTokenizer()

    phrases: Dict[str, Set[Tuple[str, ...]]] = {
        "PROD": set(),
        "PER": set(),
        "PAYM": set(),
        "LOC": set(),
        "CRIT": set(),
    }

    for sample in train_sample_it:

        text = sample["text"]

        for label in phrases:
            phrases[label] |= {
                tuple(  # pylint: disable=consider-using-generator
                    [t.text for t in tokenizer.tokenize(text[l[0] : l[1]].lower())]
                )
                for l in sample["entities"]
                if l[2] == label
            }

    patterns = []

    for label, phrase_set in phrases.items():
        for phrase in phrase_set:
            patterns += [{"label": label, "pattern": create_phrase_pattern(phrase)}]

    return patterns


@click.command()
@click.argument(
    "data_paths",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    nargs=-1,
)
@click.argument(
    "output_path",
    type=click.Path(
        dir_okay=False,
        file_okay=True,
    ),
    nargs=1,
)
def create_pattern_file(data_paths: str, output_path: str):
    # Create directory if necessary
    parant_dir = Path(output_path).parent
    if not parant_dir.exists():
        print(f"Output directory {parant_dir} does not exist. Try to create directory.")
        parant_dir.mkdir(parents=True, exist_ok=True)
        print("Successfully created")
    # Finished dir setup
    train_sample_it = []
    for data_path in data_paths:
        train_sample_it += list(srsly.read_jsonl(data_path))

    patterns = create_patterns(train_sample_it)

    srsly.write_jsonl(output_path, patterns)


if __name__ == "__main__":
    create_pattern_file()  # pylint: disable=no-value-for-parameter
