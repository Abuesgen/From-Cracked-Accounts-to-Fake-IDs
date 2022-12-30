"""
This module contains commands for evaluation of different embeddings
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from numpy.ma import MaskedArray
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances, silhouette_score

from profile_extraction.evaluation.util import embed_products
from profile_extraction.profile_clustering.embedding import get_embedder
from profile_extraction.profile_clustering.main import EMBEDDING_TYPES


@click.command()
@click.argument("products_path", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument("output_aff", type=click.Path(file_okay=True, dir_okay=False))
@click.argument("output_diff", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--embeddings", required=True, help="embeddings to use e.g. path to model")
@click.option("--embeddings-type", type=click.Choice(EMBEDDING_TYPES), required=True, help="embeddings type to use")
@click.option(
    "--plot-affinity",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
    help="Whether to plot affinity matrix",
)
@click.option(
    "--plot-differences",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
    help="Whether to plot affinity differences to visualize separability",
)
def cmd_evaluate_embeddings(  # pylint: disable=too-many-locals
    products_path: Union[str, Path],
    embeddings: str,
    embeddings_type: str,
    output_aff: str,
    output_diff: str,
    plot_affinity: Optional[str],
    plot_differences: Optional[str],
):
    """
    Evaluates embeddings using equivalence classes for products and computing their affinity matrices

    :param plot_differences: Whether to plot affinity differences to visualize separability
    :param products_path: path containing several text files representing product classes one product per line
    :param embeddings: embeddings to use e.g. path to model
    :param embeddings_type: embeddings type to use
    :param output_aff: path (CSV) to write the affinity map to
    :param output_diff: path (CSV) to write the affinity differences map to
    :param plot_affinity: Whether to plot affinity matrix
    """
    products_path = Path(products_path)

    embedder = get_embedder(embeddings, embeddings_type)

    files_products: Dict[str, List[NDArray]] = defaultdict(lambda: [])
    for file in products_path.iterdir():
        if file.is_file() and file.name.endswith(".txt"):
            with file.open(encoding="utf-8") as file_pointer:
                files_products[file.name[0:-4].replace("_", " ").title()] = embed_products(
                    file_pointer.readlines(), embedder
                )

    overall_score = compute_silhouette_score(files_products)
    print(f"Silhouette Score: {overall_score:.2f}")

    print("Cohesion for each product file (Higher is better):")
    for product_class, values in files_products.items():
        print(f"{product_class}: {get_cohesion(values):4.2f}")

    print()
    print("Coupling between product files (Lower is better):")
    affinity_table = compute_affinity_matrix(files_products)
    print(affinity_table.to_string(float_format=lambda x: f"{x:03.2f}"))
    affinity_table.to_csv(output_aff)

    difference_table = compute_differences(affinity_table)
    difference_table.to_csv(output_diff)

    if plot_differences is not None:
        difference_table_heatmap(difference_table, plot_differences)

    if plot_affinity is not None:
        affinity_table_heatmap(affinity_table, plot_affinity)


def compute_differences(affinity_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the affinity difference to the affinity matrix diagonal
    :param affinity_dataframe: DataFrame conatining affinity information
    :return: DataFrame containing a difference table
    """
    affinity = affinity_dataframe.to_numpy()
    rows, cols = affinity.shape

    differences = np.zeros(affinity.shape)
    for row in range(rows):
        for col in range(row, cols):
            differences[row, col] = affinity[row, col] - affinity[row, row]

    for col in range(cols):
        for row in range(col, rows):
            differences[row, col] = affinity[row, col] - affinity[col, col]

    mask = np.diag(np.diag(np.ones(affinity.shape)))
    differences_masked: MaskedArray = np.ma.masked_array(differences, mask=mask)
    line_mean = differences_masked.mean(axis=1)
    column_mean = np.append(np.array(differences_masked.mean(axis=0)), differences_masked.mean())
    variance = np.append(differences_masked.var(axis=1), differences_masked.var())

    print(pd.DataFrame(line_mean, index=affinity_dataframe.index).to_string(float_format=lambda x: f"{x:03.2f}"))

    differences_dataframe = pd.DataFrame(
        differences, index=affinity_dataframe.index, columns=affinity_dataframe.columns
    )
    differences_dataframe["Mean"] = line_mean
    differences_dataframe.loc["Mean"] = column_mean
    differences_dataframe["variance"] = variance

    return differences_dataframe


def compute_affinity_matrix(files_products: Dict[str, List[NDArray]]) -> pd.DataFrame:
    """
    Computes a full affinity matrix between product classes
    :param files_products: mapping from product classes to a list of embeddings
    :return: Dataframe containing affinity between product classes
    """
    affinity_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
    for product_class_one, values_one in files_products.items():
        for product_class_two, values_two in files_products.items():
            if product_class_one == product_class_two:
                affinity = get_cohesion(values_one)
            else:
                affinity = get_coupling(values_one, values_two)
            affinity_matrix[product_class_one][product_class_two] = affinity

    affinity_table = pd.DataFrame(affinity_matrix)
    return affinity_table


def compute_silhouette_score(files_products: Dict[str, List[NDArray]]) -> float:
    """
    Computes a silhouette score to show how well the given classes are separated
    A low overall Score does not imply unfit embeddings as e.g. "Uhren" (watches)
    are mostly "Elektronik" (electronic devices) => a high similarity between these classes is expected

    :param files_products: Mapping of classes to List of embeddings
    :return: computed silhouette score
    """
    cluster = []
    vectors = []
    i = 0
    for _, current_vectors in files_products.items():
        vectors += current_vectors
        cluster += len(current_vectors) * [i]
        i += 1
    overall_score: float = silhouette_score(vectors, cluster, metric="cosine")
    return overall_score


def affinity_table_heatmap(affinity_table: pd.DataFrame, file: str):
    """
    Creates a heatmap plot from an affinity table
    :param affinity_table: DataFrame containing affinity scores for product classes
    :param file: path to write plot to
    """
    pio.kaleido.scope.mathjax = None
    fig = px.imshow(
        affinity_table, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale=["#40B0A6", "#E1BE6A"], aspect="auto"
    )
    fig.update_coloraxes(showscale=False)
    fig.layout.font.family = "Times New Roman"
    fig.layout.font.size = 9
    fig.update_layout(autosize=True, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False)
    fig.write_image(file=file)


def difference_table_heatmap(difference_table: pd.DataFrame, file: str):
    """
    Creates a heatmap plot from an difference table
    :param difference_table: DataFrame containing difference scores for product classes
    :param file: path to write plot to
    """
    to_plot = difference_table.loc[:, difference_table.columns != "variance"]
    pio.kaleido.scope.mathjax = None
    fig = px.imshow(
        to_plot,
        text_auto=".2f",
        zmin=-0.3,
        zmax=0.3,
        color_continuous_scale=["#40B0A6", "white", "#E1BE6A"],
        aspect="auto",
    )
    fig.update_xaxes(type="category", color="#000000")
    fig.update_yaxes(type="category", color="#000000")
    fig.update_coloraxes(showscale=False)
    fig.update_traces(showscale=False)
    fig.add_vline(x=len(to_plot.index) - 1.5)
    fig.add_hline(y=len(to_plot.columns) - 1.5)
    fig.layout.font.family = "Times New Roman"
    fig.layout.font.size = 16
    fig.update_layout(autosize=True, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False)
    fig.write_image(file=file)


def get_coupling(products_one: List[NDArray], products_two: List[NDArray]) -> float:
    """
    Computes coupling between different product classes.
    Semantic unrelated classes should have low coupling.

    :param products_one: embeddings of the first product class
    :param products_two: embeddings of the second product class
    :return: mean of the pairwise cosine distances between products
    """
    distances: NDArray = pairwise_distances(
        X=np.row_stack(products_one), Y=np.row_stack(products_two), metric="cosine", n_jobs=-1
    )
    return 1.0 - float(distances.mean())


def get_cohesion(products: List[NDArray]) -> float:
    """
    Computes inner affinity of products.
    Works similar to get_coupling, but ignores lower triangular entries
    (including the main diagonal) to filter distance to the given product itself

    :param products: product embeddings for affinity calculation
    :return: float representing the computed affinity
    """
    distances = pairwise_distances(X=np.row_stack(products), metric="cosine", n_jobs=-1)
    mask = np.tril(np.ones((len(products), len(products))), 0)
    masked_distances: NDArray = np.ma.masked_array(distances, mask=mask)

    return 1.0 - float(masked_distances.mean())


if __name__ == "__main__":
    cmd_evaluate_embeddings()  # pylint: disable=no-value-for-parameter
