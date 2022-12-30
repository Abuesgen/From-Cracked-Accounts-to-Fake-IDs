"""
THis module creates boxplots for streaming offers
"""
# pylint: disable=invalid-name,redefined-builtin
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from profile_extraction.metadata_analysis.util import COLORS, update_fig_layout


def boxplot_streaming_offers(offerings_path: Path) -> go.Figure:
    """
    Creates a boxplot of streaming offers
    Args:
        offerings_path: path to the offerings file

    Returns:
        plotly boxplot
    """

    df = pd.read_json(offerings_path)

    fig = go.Figure()

    for i, type in enumerate(df["type"].unique()):
        df_filtered = df[df["type"] == type]
        df_filtered = df_filtered[df_filtered["crit"] != "None"]
        df_filtered = df_filtered[df_filtered["crit"] != "lifetime"]
        df_filtered = df_filtered[df_filtered["price_per_month"] < 25]
        df_filtered = df_filtered[df_filtered["price"] < 100]
        fig.add_trace(go.Box(y=df_filtered["price_per_month"], name=type + "-PPM", line=dict(color=COLORS[i])))
        fig.add_trace(go.Box(y=df_filtered["price"], name=type, line=dict(color=COLORS[i])))

    update_fig_layout(fig)
    return fig
