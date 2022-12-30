"""
This module provides helper for the creation of plots
"""
import plotly.express as px
import plotly.graph_objects as go

COLORS = px.colors.qualitative.Plotly


def update_fig_layout(fig: go.Figure):
    """
    Updates the font and margins of plots for inclusion in our paper
    Args:
        fig: figure to process
    """
    fig.update_layout(
        font_family="Times New Roman",
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", size=24, color="black"),
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
