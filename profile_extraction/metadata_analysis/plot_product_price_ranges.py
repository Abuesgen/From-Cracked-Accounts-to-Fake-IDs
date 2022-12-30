"""
Module for plotting product-price-ranges
"""
# pylint: disable=bare-except,missing-function-docstring,
import re
from importlib.resources import path

import plotly.graph_objects as go
import plotly.io as pio

import profiles
from profile_extraction.profile_creation.profile import (
    ProfileCollection,
    SummaryComponent,
)

pio.kaleido.scope.mathjax = None


def normalize_product_name(name: str):
    """
    Normalizes a product name string
    """
    step1 = name.lower().replace("  ", " ")
    normalized = re.sub(r"[^A-Za-z ]", "", step1)
    return normalized.strip()


def is_float(value_to_check):
    """
    Checks whether a given value can be interpreted as float
    """
    try:
        float(value_to_check)
        return True
    except:
        return False


def normalize_price(name: str):
    normalized = re.sub(r"[^0-9 \.,]", "", name).replace(",", ".")
    if normalized.endswith("."):
        normalized = normalized[:-1]

    if is_float(normalized):
        return float(normalized)

    return -1


if __name__ == "__main__":
    with path(profiles, "") as profile_dir:

        all_priced_products = []

        for file_path in profile_dir.iterdir():

            if not (file_path.is_file() and file_path.suffix == ".json"):
                continue

            profiles = ProfileCollection.parse_file(file_path).profiles  # type: ignore

            summaries = []
            for profile in profiles:  # type: ignore
                summaries += [
                    component for component in profile.components if isinstance(component, SummaryComponent)
                ]

            all_priced_products += [
                (normalize_product_name(p.product), [normalize_price(pr) for pr in p.price])
                for s in summaries
                for p in s.products
                if len(p.price) > 0
            ]

    prod_list_prices = {n: [p for (na, p) in all_priced_products if n == na] for (n, p) in all_priced_products}

    prod_prices = {n: [e for pl in p for e in pl if e >= 0] for (n, p) in prod_list_prices.items()}

    top_products = {n: ps for (n, ps) in prod_prices.items() if len(ps) > 15}

    PRODUCTS = {
        "tshirts": "T-Shitrs",
        "nord vpn": "VPN",
        "uhren": "Watches",
        "dazn": "Streamin Acc.",
        "views": "Views",
    }

    fig = go.Figure()

    for prod, value in PRODUCTS.items():
        fig.add_trace(go.Box(y=[p for p in top_products[prod] if p < 1000], name=value))

    fig.update_layout(
        yaxis_title="Estimated Price in €",
        font_family="Times New Roman",
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", size=24, color="black"),
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
    )

    fig.write_image("price_ranges_times.pdf")

    i = 10
