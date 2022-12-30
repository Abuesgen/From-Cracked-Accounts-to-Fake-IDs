"""
Transfers the splitted text files into doccano format with unique ids for a product.
Thus, we can compute iaa.
"""
import os
from pathlib import Path
from typing import Dict, List

import srsly


def replace(value: str):
    """
    Replace all new lines + leading and trailing whitespaces
    """
    return value.replace("\n", "").strip()


def read_annotations(path: str) -> Dict[str, List[str]]:
    """
    Reads all text files and stores them in a dictionary.
    The text file's name represents the label.
    """
    result = {}
    for text_file in os.listdir(path):
        with open(Path(path) / text_file, "r", encoding="UTF-8") as input_file:
            result[text_file] = [replace(t) for t in input_file.readlines()]

    return result


def transform_to_rap(products_per_label: Dict[str, List[str]], all_products: List[str]):
    """
    Uses the products_per_label mapping to build a file conforming the doccano format.
    all_products facilitate the unique id for a product.
    """
    rap = []
    for file_name_w_ext in products_per_label:
        label = file_name_w_ext.split(".")[0]
        for text in products_per_label[file_name_w_ext]:
            idx = all_products.index(replace(text))

            rap.append({"id": idx, "text": text, "label": [label]})

    return rap


if __name__ == "__main__":
    with open("all_product.txt", "r", encoding="UTF-8") as file:
        products = [replace(t) for t in file.readlines()]

    for file_name in ["clustering_andre", "clustering_lars", "clustering_phil"]:
        annotations = read_annotations(file_name)
        rap_version = transform_to_rap(annotations, products)

        srsly.write_jsonl(file_name + ".jsonl", rap_version)
