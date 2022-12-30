"""
This module contains code for evaluation of the Nearest Neighbour PROD-PRICE relation
"""
import json
from pathlib import Path
from typing import Dict, List, Union

import click
from pydantic import BaseModel, validator


class JsonLabel(BaseModel):
    """
    Represents a SpanLabel in doccano JSONL format
    """

    start_idx: int
    end_idx: int
    label: str


class JsonRelationLabel(BaseModel):
    """
    Represents a RelationLabel in doccano JSONL format
    """

    start_idx_one: int
    end_idx_one: int
    start_idx_two: int
    end_idx_two: int
    label: str


class JsonDatasetEntry(BaseModel):
    """
    Represents a document in doccano JSONL format
    """

    text: str
    entities: List[Union[List[Union[str, int]], JsonLabel]] = []
    relations: List[Union[List[Union[str, int]], JsonRelationLabel]] = []

    @validator("entities")
    def parse_entitiess(cls, value):  # pylint: disable=no-self-argument
        """
        Converts doccanos list style entitiess into DTOS
        :param value: entities values
        :return: Jsonentities
        """
        new_labels = []

        for label in value:
            if isinstance(label, List):
                new_labels.append(JsonLabel(start_idx=label[0], end_idx=label[1], label=label[2]))
            else:
                new_labels.append(label)

        return new_labels

    @validator("relations")
    def parse_relation_labels(cls, value):  # pylint: disable=no-self-argument
        """
        Converts doccanos list style relations into DTOS
        :param value: label values
        :return: JsonRelationLabel
        """
        new_labels = []

        for label in value:
            if isinstance(label, List):
                if label[4] == "PROD_PRICE":
                    new_labels.append(
                        JsonRelationLabel(
                            start_idx_one=label[0],
                            end_idx_one=label[1],
                            start_idx_two=label[2],
                            end_idx_two=label[3],
                            label=label[4],
                        )
                    )
            else:
                new_labels.append(label)

        return new_labels


@click.command()
@click.option("--dataset", type=click.Path(file_okay=True, dir_okay=False, exists=True), required=True)
def cmd_eval_nn_results(dataset: Union[str, Path]):
    """
    Evaluates a given dataset containing PROD-PRICE relations against the nearest neighbour algorithm
    :param dataset: dataset containing annotated PROD-PRICE ranges
    """
    dataset = Path(dataset)

    with dataset.open(encoding="utf-8") as entries:
        dataset_entries = [JsonDatasetEntry.parse_raw(line) for line in entries]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for entry in dataset_entries:
        cur_tp, cur_fp, cur_fn = count_correct_prod_price_relations(entry)
        true_positives += cur_tp
        false_positives += cur_fp
        false_negatives += cur_fn

    metrics: Dict[str, float] = {
        "f_1": true_positives / (true_positives + 0.5 * (false_positives + false_negatives)),
        "precision": true_positives / (true_positives + false_positives),
        "recall": true_positives / (true_positives + false_negatives),
    }

    print(json.dumps(metrics))


def count_correct_prod_price_relations(datapoint: JsonDatasetEntry):
    """
    Calculates true positives, false positives and false negatives in PROD-PRICE relation creation using the
    nearest neighbour algorithm
    :param datapoint: Single document containing annotated products, prices and product-price relations
    """
    prods = [prod for prod in datapoint.entities if isinstance(prod, JsonLabel) and prod.label == "PROD"]
    money = [money for money in datapoint.entities if isinstance(money, JsonLabel) and money.label == "MONEY"]

    found_prices = [find_price(prod, money) for prod in prods]
    found_prices = [price for price in found_prices if price is not None]
    correct_prices = [relation for relation in datapoint.relations if isinstance(relation, JsonRelationLabel)]

    found_correct_prices = 0
    not_found_correct_prices = 0
    for correct_price in correct_prices:
        # check both relation directions
        reversed_correct = JsonRelationLabel(
            start_idx_one=correct_price.start_idx_two,
            end_idx_one=correct_price.end_idx_two,
            start_idx_two=correct_price.start_idx_one,
            end_idx_two=correct_price.end_idx_one,
            label=correct_price.label,
        )
        if correct_price in found_prices:
            found_prices.pop(found_prices.index(correct_price))
            found_correct_prices += 1
        elif reversed_correct in found_prices:
            found_prices.pop(found_prices.index(reversed_correct))
            found_correct_prices += 1
        else:
            not_found_correct_prices += 1

    # tp, fp, fn
    return found_correct_prices, len(found_prices), not_found_correct_prices


def find_price(prod: JsonLabel, money: List[JsonLabel]):
    """
    Find price for product using the nearest neighbour approach
    :param prod: product to search price for
    :param money: annotated money entities
    :return: RelationLabel if a price could be found
    """
    distance = 200
    price = None

    for current_price in money:
        current_dist = abs(prod.end_idx - current_price.start_idx)
        if current_dist < distance:
            distance = current_dist
            price = current_price

    return (
        JsonRelationLabel(
            start_idx_one=prod.start_idx,
            end_idx_one=prod.end_idx,
            start_idx_two=price.start_idx,
            end_idx_two=price.end_idx,
            label="PROD_PRICE",
        )
        if price
        else None
    )
