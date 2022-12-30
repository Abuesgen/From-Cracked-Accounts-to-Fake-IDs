"""
This module is used to create plots for our case study
"""

from pathlib import Path
from typing import List

import typer

from profile_extraction.metadata_analysis.box_plot_streamining_offers import (
    boxplot_streaming_offers,
)
from profile_extraction.metadata_analysis.create_relation_based_product_offerings import (
    create_offerings,
)
from profile_extraction.metadata_analysis.hist_constr_type import (
    histogram_of_subscription_lengths,
)
from profile_extraction.metadata_analysis.price_per_month_and_length_of_crit import (
    price_scatter_subscription_period,
)

app = typer.Typer()


@app.command()
def create_offerings_file(output_path: Path, profile_paths: List[Path]):
    """
    Extracts product offerings from chat exports
    """
    create_offerings(output_path, profile_paths)


@app.command()
def create_boxplot_offers(offers_path: Path, image_path: Path):
    """
    Creates a boxplot of price ranges for products
    Args:
        offers_path: Path to a json file containg product offerings
        image_path: path to save the figure to
    """
    fig = boxplot_streaming_offers(offers_path)
    fig.write_image(image_path)


@app.command()
def create_subscriptions_hist(offers_path: Path, image_path: Path):
    """
    Creates a histogram of subscription lengths for different product types
    Args:
        offers_path: Path to a json file containg product offerings
        image_path: path to save the figure to
    """
    fig = histogram_of_subscription_lengths(offers_path)
    fig.write_image(image_path)


@app.command()
def create_cost_scatterplot(offers_path: Path, image_path: Path):
    """
    Creates a scatterplot showing cost-per-month for subscriptions
    Args:
        offers_path: Path to a json file containg product offerings
        image_path: path to save the figure to
    """
    fig = price_scatter_subscription_period(offers_path)
    fig.write_image(image_path)


def main():
    """
    Main entrypoint for typer cli
    """
    app()
