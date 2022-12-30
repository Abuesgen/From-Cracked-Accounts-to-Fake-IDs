"""
Module to filter examples containing prod price relaitons
"""
from pathlib import Path
from typing import List, Optional

import srsly
import typer


# pylint: disable=dangerous-default-value
def cmd(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    filter_labels: List[str] = ["PROD", "MONEY"],
    label_column: Optional[str] = "labels",
):
    """
    Splits a given file into two files. One containing all Examples having ALL filter_labels present and one file
    with all other examples
    """
    filter_labels_set = set(filter_labels)
    entries_with_matching_labels = []
    entries_others = []
    for entry in srsly.read_jsonl(input_file):
        labels = entry[label_column]
        present_labels = [e[2] for e in labels]

        if filter_labels_set.issubset(set(present_labels)):
            entries_with_matching_labels.append(entry)
        else:
            entries_others.append(entry)

    file_with_matching_labels = output_dir / f"{input_file.stem}_{'_'.join(list(filter_labels))}.jsonl"
    file_others = output_dir / f"{input_file.stem}_others.jsonl"

    srsly.write_jsonl(file_with_matching_labels, entries_with_matching_labels)
    srsly.write_jsonl(file_others, entries_others)


def main():
    """
    dummy main for runscript
    """
    typer.run(cmd)  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
