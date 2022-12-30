"""
This module creates a histogram of subscription lengths
"""

# pylint: disable=invalid-name
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from profile_extraction.metadata_analysis.util import update_fig_layout


def histogram_of_subscription_lengths(offerings_path: Path) -> go.Figure:
    """
    Creates a plotly histogram of subscription lengths of product offerings
    Args:
        offerings_path: path to the offerings file

    Returns:
        created figure

    """
    df = pd.read_json(offerings_path)

    df_filtered = df.query(
        "crit == 'None' or crit == 'lifetime' or crit == 1 or crit == 12 or crit == 24 or crit == 36"
    )
    df_filtered = df_filtered.query("type == 'NordVPN' or type == 'Netflix' or type == 'Sky'")

    df_filtered["str_crit"] = [crit.replace("None", "unknown") for crit in df_filtered["crit"].astype(str)]

    fig = px.histogram(
        df_filtered,
        x="str_crit",
        color="type",
        barmode="group",
        histnorm="probability density",
        category_orders={"str_crit": ["1", "12", "24", "36", "lifetime", "unknown"]},
        labels={"str_crit": "Subscription length", "type": "Product"},
    )
    fig.update_yaxes(title="offerings (percentage)")

    update_fig_layout(fig)

    return fig
