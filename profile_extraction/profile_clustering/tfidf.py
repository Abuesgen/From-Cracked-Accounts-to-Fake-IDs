"""
This module clusters profiles using a simple TfIDf vectorizer without word embeddings
"""
import re
import typing
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

import click
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from profile_extraction.profile_clustering.cluster import (
    EMPTY_PROFILE,
    NO_CLUSTER,
    compute_silhouette_score,
)
from profile_extraction.profile_clustering.main import write_results
from profile_extraction.profile_creation.profile import (
    ProductComponent,
    Profile,
    ProfileCollection,
)


@click.command()
@click.option("--profiles-path", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option("--visualizations-path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--output-path", type=click.Path(dir_okay=True, file_okay=False))
@click.option(
    "--threshold",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.1,
    help="Max cosine distance between two clusters to merge. Lower values are equal to stricter clustering.",
)
def cmd_cluster_tf_idf(
    profiles_path: Union[str, Path],
    visualizations_path: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float,
):
    """
    Command to cluster profiles
    :param profiles_path: path to profile json
    :param visualizations_path: path to visualizations
    :param output_path: path to write clusters to
    :param threshold: threshold for agglomerative clustering
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    profiles = ProfileCollection.parse_file(profiles_path)
    profile_list = list(profiles)

    vectorizer = EasyTfIdfVectorizer(profiles)

    clustered_profiles, score = cluster_profiles(profile_list, vectorizer, threshold)
    names = find_cluster_names(clustered_profiles, vectorizer)
    clustered_profiles["cluster"] = clustered_profiles["cluster"].map(names)

    write_results(clustered_profiles, Path(visualizations_path), Path(output_path), score)


def find_cluster_names(clustered_profiles: pd.DataFrame, vectorizer):
    """
    finds cluster names
    :param clustered_profiles: dataframe containing clustered profiles
    :param vectorizer: TFIDF vectorizer to use
    :return: dictionary mapping cluster number to a name
    """
    name_map: Dict[int, str] = {-1: NO_CLUSTER, -2: EMPTY_PROFILE}

    distinct_clusters = set(clustered_profiles["cluster"]) - {-1, -2}
    for cluster in distinct_clusters:
        name_map[cluster] = find_cluster_name(
            clustered_profiles.loc[clustered_profiles["cluster"] == cluster, :], vectorizer
        )

    return name_map


def find_cluster_name(clustered_profiles: pd.DataFrame, vectorizer):
    """
    generates a cluster name from a single given cluster
    :param clustered_profiles: dataframe containing clustered profiles
    :param vectorizer: TFIDF vectorizer to use
    :return: the name for the given cluster
    """
    cluster_vector = np.mean(np.column_stack(clustered_profiles["vector"]), axis=1)
    product_list: List[str] = []
    for profile in clustered_profiles["profile"]:
        products = [product.product.text for product in profile.components if isinstance(product, ProductComponent)]
        product_list.extend(products)

    product_list = list(set(product_list))
    product_vectors = [
        np.array(vec).reshape(-1)
        for vec in np.vsplit(np.array(vectorizer.vectorizer.transform(product_list).todense()), len(product_list))
    ]

    best_name = "NO_NAME"
    best_dist = 1
    for product, product_vector in zip(product_list, product_vectors):
        dist = distance.cosine(product_vector, cluster_vector)
        if dist < best_dist:
            best_dist = dist
            new_name = re.sub(r"\W", "", product)
            best_name = new_name if new_name else best_name

    return best_name


def cluster_profiles(profiles, vectorizer, threshold: float):
    """
    Clusters profiles using the tf-idf vectorizer
    :param profiles: profiles to cluster
    :param vectorizer: vectorizer to use
    :param threshold: threshold for clustering algorithm
    :return: a dataframe containing clustered profiles
    """
    vectors = vectorizer.embed(profiles)

    zero_profiles = []
    non_zero_profiles = []
    non_zero_vectors = []
    for profile, vector in zip(profiles, vectors):
        if vector.any():
            non_zero_profiles += [profile]
            non_zero_vectors += [vector]
        else:
            zero_profiles += [profile]

    clusters = AgglomerativeClustering(
        n_clusters=None, affinity="cosine", linkage="average", distance_threshold=threshold
    ).fit_predict(non_zero_vectors)

    cluster_score = compute_silhouette_score(clusters, non_zero_vectors)
    print(f"Silhouette Score: {cluster_score}")
    counter: typing.Counter[int] = Counter(clusters)
    clusters = [(cluster if counter[cluster] > 3 else -1) for cluster in clusters]

    clustered_profiles = pd.DataFrame(
        {
            "profile": list(zero_profiles) + list(non_zero_profiles),
            "vector": (len(zero_profiles) * [np.zeros(len(vectorizer.vectorizer.get_feature_names_out()))])
            + list(non_zero_vectors),
            "cluster": (len(zero_profiles) * [-2]) + clusters,
        }
    )

    return clustered_profiles, cluster_score


class EasyTfIdfVectorizer:  # pylint: disable=too-few-public-methods
    """
    Vectorizes profiles using TfidfVectorizer
    """

    def __init__(self, profiles: Iterable[Profile]):
        self.vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 4), analyzer="char_wb")
        self.vectorizer.fit([self._get_doc(profile) for profile in profiles])

    def embed(self, profile: Union[Profile, List[Profile]]) -> List[NDArray]:
        """
        Embeds a set of profiles
        :param profile: profiles to embed
        :return: list of embeddings
        """
        profiles = profile if isinstance(profile, List) else [profile]
        return self._embed(profiles)

    def _embed(self, profiles: List[Profile]) -> List[NDArray]:
        profile_docs = [self._get_doc(profile) for profile in profiles]
        profile_vectors = self.vectorizer.transform(profile_docs).todense()
        vector_list = np.vsplit(profile_vectors, len(profiles))

        return [np.array(vector).reshape(-1) for vector in vector_list]

    @staticmethod
    def _get_doc(profile: Profile):
        products = [
            re.sub(r"\s", "-", product.product.text)
            for product in profile.components
            if isinstance(product, ProductComponent)
        ]
        return " ".join(products)
