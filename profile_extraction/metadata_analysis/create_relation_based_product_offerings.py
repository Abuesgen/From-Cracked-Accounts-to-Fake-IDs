"""
This module creates product offerings from the models data
"""
# pylint: disable=invalid-name,missing-function-docstring,too-many-branches,too-many-return-statements,consider-using-in
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import typer

from profile_extraction.metadata_analysis.plot_product_price_ranges import (
    normalize_price,
)
from profile_extraction.profile_creation.profile import (
    ProductComponent,
    Profile,
    ProfileCollection,
    RelationComponent,
)

app = typer.Typer()


def normalize_crit_multiplier(param):
    if param == "ein" or param == "eins":  # pylint: disable=no-else-return
        return 1
    elif param == "zwei":
        return 2
    elif param == "drei":
        return 3
    elif param == "vier":
        return 4
    elif param == "fünf" or param == "funf":
        return 5
    elif param == "sechs":
        return 6
    elif param == "sieben":
        return 7
    elif param == "acht":
        return 8
    elif param == "neun":
        return 9
    elif param == "zehn":
        return 10
    elif param == "elf":
        return 11
    elif param == "zwölf" or param == "zwolf":
        return 12
    else:
        try:
            return int(param)
        except:  # pylint: disable=bare-except)
            # this should never happen, sinze our regex would not have matched
            return 1


def normalize_product(product, price, crit):
    products = {
        "Netflix": "netfl",
        "NordVPN": "nordv",
        "Sky": "sky",
    }

    pattern = re.compile(
        r"(\d+|ein|eins|zwei|drei|vier|f[uü]nf|sechs|sieben|acht|neun|zehn|elf|zw[oö]lf)(?:\s+)?(jahr|monat|)[e]?"
    )
    results = []

    for prod_key, prod_pattern in products.items():
        if prod_pattern in product.text.lower():
            is_normalized = False

            if crit is None:
                normalized_price = normalize_price(price.text)

                results.append(
                    {
                        "product": product.text,
                        "type": prod_key,
                        "price_per_month": None,
                        "price": normalized_price,
                        "crit": "None",
                        "is_normalized": False,
                    }
                )
            else:
                # Here we have a crit annotation and try to identify time constraints
                m = pattern.search(crit.text.lower())
                if m:
                    # Here we have a constraint of the form
                    # <number> <time unit>

                    multiplier = 1

                    groups = m.groups()

                    if len(groups) == 2:
                        multiplier = 1 if groups[0] is None else normalize_crit_multiplier(groups[0])
                    else:
                        # this should not happen
                        pass

                    if "ja" in groups[1]:
                        # we have a yearly constraint
                        # in each other case we assume a monthly constraint
                        multiplier *= 12

                    normalized_price = normalize_price(price.text)

                    normalized_price /= multiplier

                    is_normalized = True

                    results.append(
                        {
                            "product": product.text,
                            "type": prod_key,
                            "price_per_month": normalized_price,
                            "price": normalize_price(price.text),
                            "crit": multiplier,
                            "is_normalized": is_normalized,
                        }
                    )
                else:
                    normalized_price = normalize_price(price.text)
                    if "jah" in crit.text.lower():
                        multiplier = 12
                        results.append(
                            {
                                "product": product.text,
                                "type": prod_key,
                                "price_per_month": normalized_price / multiplier,
                                "price": normalized_price,
                                "crit": multiplier,
                                "is_normalized": True,
                            }
                        )
                    elif "mon" in crit.text.lower():
                        results.append(
                            {
                                "product": product.text,
                                "type": prod_key,
                                "price_per_month": normalized_price,
                                "price": normalized_price,
                                "crit": 1,
                                "is_normalized": True,
                            }
                        )
                    elif "lif" in crit.text.lower():
                        results.append(
                            {
                                "product": product.text,
                                "type": prod_key,
                                "price_per_month": None,
                                "price": normalized_price,
                                "crit": "lifetime",
                                "is_normalized": False,
                            }
                        )
                    else:
                        pass
    if not results:
        pass
    return results


def normalize_old_product(prod_component):
    if "netflix" in prod_component.product.text.lower():
        if prod_component.price is None:
            return None

        price = normalize_price(prod_component.price.text)

        return {
            "product": prod_component.product.text,
            "type": "netflix",
            "price_per_month": None,
            "price": price,
            "crit": "None",
            "is_normalized": False,
        }

    return None


def create_offerings(output_path: Path, profile_paths: List[Path]):

    all_product_offerings: List[Dict[str, Any]] = []
    all_old_product_offerings: List[Dict[str, Any]] = []
    for profile_path in profile_paths:
        profiles = ProfileCollection.parse_file(profile_path)

        for profile in profiles:

            process_profile(all_old_product_offerings, all_product_offerings, profile)

    df = pd.DataFrame(all_product_offerings)

    df.to_json(output_path)


def process_profile(all_old_product_offerings, all_product_offerings, profile: Profile):
    products = [
        normalize_old_product(relation) for relation in profile.components if isinstance(relation, ProductComponent)
    ]
    all_old_product_offerings.extend([p for p in products if p is not None])
    prod_price_relations = [relation for relation in profile.components if isinstance(relation, RelationComponent)]
    all_prod_price_messages = {relation.message for relation in prod_price_relations}
    for message in all_prod_price_messages:

        message_prod_price_relations = [
            relation
            for relation in prod_price_relations
            if relation.message.id == message.id and relation.relation.label == "PROD_PRICE"
        ]

        message_price_crit_relations = [
            relation
            for relation in prod_price_relations
            if relation.message.id == message.id and relation.relation.label != "PROD_PRICE"
        ]

        for prod_price_relation in message_prod_price_relations:

            price = prod_price_relation.relation.tail

            crits_for_price = [crit for crit in message_price_crit_relations if crit.relation.head == price]

            if crits_for_price:
                for crit in crits_for_price:
                    all_product_offerings.extend(
                        normalize_product(
                            product=prod_price_relation.relation.head, price=price, crit=crit.relation.tail
                        )
                    )
            else:
                all_product_offerings.extend(
                    normalize_product(product=prod_price_relation.relation.head, price=price, crit=None)
                )
