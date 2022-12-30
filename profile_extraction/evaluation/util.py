"""
This module contains util functions for evaluation
"""
import re
import typing
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Union

import click
import numpy as np
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings
from flair.tokenization import SegtokTokenizer
from numpy.typing import NDArray
from sklearn.cluster import OPTICS

from profile_extraction.profile_clustering.embedding import (
    ProfileEmbedder,
    batch,
    get_embedder,
)
from profile_extraction.profile_clustering.main import EMBEDDING_TYPES
from profile_extraction.profile_creation.profile import (
    ProfileCollection,
    SummaryComponent,
)


def get_all_products(profiles: ProfileCollection) -> Set[str]:
    """
    Extracts and normalizes all Products in a ProfileCollection
    :param profiles: ProfileCollection containing Profiles with products
    :return: A set of normalized product names
    """

    product_names: Set[str] = set([])
    for profile in profiles:
        for component in profile.components:
            if isinstance(component, SummaryComponent):
                product_texts = {product.product for product in component.products}
                product_texts = {re.sub(r"\W", " ", product) for product in product_texts}
                product_texts = {re.sub(r"\s+", " ", product) for product in product_texts}
                product_texts = {product.strip().lower() for product in product_texts}
                product_texts = {product for product in product_texts if len(product) > 1}

                product_names.update(product_texts)
    return product_names


def embed_products(products: List[str], embedder: TokenEmbeddings) -> List[NDArray]:
    """
    Embeds a List of product strings using a given embedder
    :param products: products to embed
    :param embedder: embedder to use
    :return: List of product embeddings
    """
    sentences = [Sentence(prod, use_tokenizer=SegtokTokenizer()) for prod in products]

    for mini_batch in batch(sentences, 16):
        embedder.embed(mini_batch)

    product_vectors = [
        np.mean(np.column_stack([token.embedding.detach().cpu().numpy() for token in sent]), axis=1)
        for sent in sentences
    ]
    return product_vectors


def cluster_products(
    products: List[str], embedder: TokenEmbeddings, threshold: float, min_size: int
) -> Dict[int, List[str]]:
    """
    Clusters a given product list using given embeddings
    :param products: products to cluster
    :param embedder: embeddings to use for clustering
    :return: dictionary mapping a cluster number to its products
    """
    product_vectors = embed_products(products, embedder)

    assert product_vectors[0].shape == (embedder.embedding_length,)

    clusters = OPTICS(
        cluster_method="dbscan", max_eps=threshold, n_jobs=-1, metric="cosine", min_samples=2
    ).fit_predict(np.row_stack(product_vectors))

    assert len(clusters) == len(products)

    clustered_products: Dict[int, List[str]] = defaultdict(lambda: [])
    for product, cluster in zip(products, clusters):
        clustered_products[cluster] += [product]

    counter: typing.Counter[int] = Counter(clusters)
    for cluster, count in counter.items():
        if count < min_size:
            clustered_products[-1] += clustered_products[cluster]
            clustered_products.pop(cluster)

    return clustered_products


@click.command()
@click.argument("profiles_path", type=click.Path(file_okay=True, dir_okay=False, exists=True))
def cmd_list_all_products(profiles_path: str):
    """
    Prints a products of a given ProfileCollection to stdout
    :param profiles_path: path to a profile jsong
    """
    profiles = ProfileCollection.parse_file(profiles_path)

    for product in get_all_products(profiles):
        print(product)


@click.command()
@click.argument("products_path", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option("--embeddings", required=True)
@click.option("--embeddings-type", type=click.Choice(EMBEDDING_TYPES), default=ProfileEmbedder.MODEL)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.1,
    help="distance threshold for clustering",
)
@click.option("--min-size", type=click.IntRange(min=1), default=10)
def cmd_cluster_products(  # pylint: disable=too-many-locals
    products_path: Union[str, Path],
    output_dir: Union[str, Path],
    embeddings: str,
    embeddings_type: str,
    threshold: float,
    min_size: int,
):
    """
    Util function to use pre-clustering on products.
    :param threshold: distance threshold for clustering
    :param products_path: Path to a text file containing one product per line
    :param output_dir: dir to write clusterd produts to
    :param embeddings: Embeddings to use
    :param embeddings_type: Embeddings type to use (see profile-clustering)
    """
    products_path = Path(products_path)
    output_dir = Path(output_dir)

    embedder = get_embedder(embeddings, embeddings_type)

    with products_path.open(encoding="utf-8") as products_file:
        products = products_file.readlines()

    clustered_products = cluster_products(products, embedder, threshold, min_size)

    for key, value in clustered_products.items():
        value.sort()
        cluster_file = output_dir / f"{key:03d}_{len(value):04d}.txt"
        with cluster_file.open("w", encoding="utf-8") as file:
            for prod in value:
                print(prod.strip(), file=file)
