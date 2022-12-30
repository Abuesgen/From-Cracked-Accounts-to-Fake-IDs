"""
This module creates a scatterplot with logarithmic trendlines for subscripiton prices
"""
# pylint: disable=invalid-name
from pathlib import Path

import pandas as pd
import plotly.express as px

from profile_extraction.metadata_analysis.util import update_fig_layout


def price_scatter_subscription_period(offers_path: Path):
    """
    Creates a scatterplot of subscription prices
    Args:
        offers_path: Path to a json dataframe containing product offerings

    Returns:
        Scatterplot of subscription prices
    """
    df = pd.read_json(offers_path)

    df = df[df["crit"] != "None"]
    df = df[df["crit"] != "lifetime"]
    df = df[df["crit"] < 50]
    df = df[df["price_per_month"] < 25]

    fig = px.scatter(
        df,
        x="crit",
        y="price_per_month",
        trendline="ols",
        trendline_options=dict(log_x=True),
        color="type",
        labels={"type": "Product"},
        category_orders={"crit": list(range(1, 51))},
    )
    fig.update_xaxes(title="Length of subscription (months)")
    fig.update_yaxes(title="Average price per month (€)")

    update_fig_layout(fig)
    return fig
