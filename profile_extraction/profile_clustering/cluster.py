"""
This module provides functions for profile clustering
"""
import logging
import typing
from collections import Counter, defaultdict
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from flair.data import Sentence
from flair.embeddings.token import TokenEmbeddings
from flair.tokenization import SegtokTokenizer
from googletrans import Translator
from numpy.typing import NDArray
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from profile_extraction.profile_clustering.embedding import ProfileEmbedder, batch
from profile_extraction.profile_creation.profile import (
    ProductComponent,
    Profile,
    ProfileCollection,
)

log = logging.getLogger(__name__)

NO_CLUSTER = "Kein Cluster"
EMPTY_PROFILE = "Keine Produkte"

COLORS = [
    "#8bfb32",
    "#5700b2",
    "#1db000",
    "#ec43f5",
    "#6bff81",
    "#fb77ff",
    "#00d264",
    "#ff49a3",
    "#d0ff74",
    "#00389e",
    "#ffe747",
    "#b587ff",
    "#afbd00",
    "#61005a",
    "#fbffa0",
    "#00193e",
    "#d99e00",
    "#0177c9",
    "#ce7700",
    "#17c7ff",
    "#ff7946",
    "#02d7c1",
    "#b60072",
    "#01944b",
    "#ae0033",
    "#00ad7b",
    "#410026",
    "#a6ffda",
    "#590009",
    "#ade9ff",
    "#7a4200",
    "#a5baff",
    "#497f00",
    "#ffc7e6",
    "#005c18",
    "#ff887c",
    "#004854",
    "#967600",
    "#ffc0a2",
    "#243a00",
]


def tsne_plot(clustered_profiles: pd.DataFrame):  # pylint: disable=too-many-locals
    """
    Creates a TSNE plot to visualize the generated profile cluster

    :param clustered_profiles: DataFrame containing all clustered Profiles
    :return: a Plotly figure of an annotated TSNE-plot
    """

    to_show = clustered_profiles.loc[clustered_profiles["cluster"] != NO_CLUSTER, :]
    to_show = to_show.loc[clustered_profiles["cluster"] != EMPTY_PROFILE, :]
    to_show.loc[:, "cluster"] = to_show.loc[:, "cluster"].astype(str)

    features = np.row_stack(to_show.loc[:, "vector"])
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        n_iter=1000000,
        method="exact",
        n_iter_without_progress=300000,
        n_jobs=-1,
        init="pca",
        learning_rate="auto",
        random_state=17,
        perplexity=min(len(features) - 1, 30),
    )

    clusters = list(to_show.loc[:, "cluster"])
    projections = tsne.fit_transform(features)

    projection_to_cluster: Dict[str, typing.List[NDArray]] = defaultdict(lambda: [])
    for key, coord in zip(clusters, projections):
        projection_to_cluster[key].append(coord)

    cluster_mids: Dict[str, NDArray] = {}
    for key, value in projection_to_cluster.items():
        coord = np.median(np.vstack(value), axis=0)
        cluster_mids[key] = coord

    dists: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
    for key_one, coord_one in cluster_mids.items():
        for key_two, coord_two in cluster_mids.items():
            dists[key_one][key_two] = float(np.linalg.norm(coord_two - coord_one))

    fig = px.scatter(
        projections,
        x=0,
        y=1,
        color=clusters,
        color_discrete_sequence=COLORS,
        labels={"color": ""},
    )

    for key, coord in cluster_mids.items():
        y_shift = 0
        for key2, dist in dists[key].items():
            if dist < 1 and key2 != key:
                print(key, ",", key2, ",", dist)
                y_shift += 20
                dists[key2][key] = 100

        fig.add_annotation(
            x=coord[0],
            y=coord[1],
            text=key,
            showarrow=(y_shift != 0),
            arrowhead=1,
            font=dict(color="black", size=14),
            bgcolor="rgba(255,255,255,0.75)",
            ay=-y_shift,
        )
    fig.update_xaxes(title="", showticklabels=False, visible=False)
    fig.update_yaxes(title="", showticklabels=False, visible=False)
    fig.layout.font.family = "Times New Roman"
    fig.update_layout(autosize=True, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_layout(showlegend=False)
    return fig


def find_cluster_names(clustered: pd.DataFrame, embedder: TokenEmbeddings) -> Dict[int, str]:
    """
    Generates cluster names for a given clustered ProfileCollection
    :param clustered: clustered profiles to find names for
    :param embedder: embedder used for clustering
    :return: a mapping of cluster to name
    """
    distinct_clusters = set(clustered.loc[:, "cluster"]) - {-1, -2}
    cluster_names: Dict[int, str] = {-1: NO_CLUSTER, -2: EMPTY_PROFILE}
    for cluster in tqdm(distinct_clusters, desc="Generating cluster names"):
        rows = clustered.loc[clustered["cluster"] == cluster, :]
        cluster_names[cluster] = find_cluster_name(rows.loc[:, "profile"], rows.loc[:, "vector"], embedder)
    return cluster_names


def find_cluster_name(profiles: Iterable[Profile], profile_vectors, embedder: TokenEmbeddings) -> str:
    """
    Tries to find a representative product for a cluster of profiles. Using the mean of all profile vectors the nearest
    (cosine distance) product is used as cluster name

    :param profiles: Profiles to find a representative name for
    :param profile_vectors:  vector representations of the profiles
    :param embedder: embedder to create word embeddings for found products
    :return: Proposed Name for the given cluster
    """

    # Calculate representative vector for all profiles
    rep_vec = np.mean(np.column_stack(profile_vectors), axis=1)

    # Get all Product names and their embeddings
    products = []
    for profile in profiles:
        products += [component for component in profile.components if isinstance(component, ProductComponent)]

    product_names = list({product.product.text for product in products})
    product_sentences = [Sentence(product, use_tokenizer=SegtokTokenizer()) for product in product_names]

    batch_size = 1
    for current_batch in batch(product_sentences, batch_size):
        embedder.embed(current_batch)

    # Calculate distance of each product to the represenatative vector. The closest ist chosen as name
    best_name = ""
    best_dist = 1
    for sentence in product_sentences:
        prod_embedding = np.mean(np.column_stack([t.embedding.detach().cpu().numpy() for t in sentence]), axis=1)
        dist = distance.cosine(prod_embedding, rep_vec)
        if dist < best_dist:
            best_dist = dist
            best_name = sentence.to_original_text()

    return str(Translator().translate(best_name, src="de", dest="en").text)


def cluster_profiles(  # pylint: disable=too-many-locals
    profiles: ProfileCollection,
    profile_embedder: ProfileEmbedder,
    min_cluster_size: int = 3,
    distance_threshold: float = 0.1,
    linkage: str = "average",
) -> typing.Tuple[pd.DataFrame, float]:
    """
    Clusters a ProfileCollection
    :param use_tf_idf: Whether to use tf-idf weighting of embeddings
    :param linkage: linkage method to use see sklearn AgglomerativeClustering
    :param distance_threshold: cluster having a distance larger than threshold will not be merged
    :param min_cluster_size: Cluster smaller than this size will be removed after clustering
    :param profiles: profiles to cluster
    """

    (non_zero_profiles, non_zero_vectors), (zero_profiles, zero_vectors) = profile_embedder.embed(profiles)

    clusters = AgglomerativeClustering(
        n_clusters=None, affinity="cosine", linkage=linkage, distance_threshold=distance_threshold
    ).fit_predict(non_zero_vectors)

    cluster_score = compute_silhouette_score(clusters, non_zero_vectors)
    print(f"Silhouette Score: {cluster_score}")
    cluster_sizes: typing.Counter[int] = Counter(clusters)
    clusters = [(cluster if cluster_sizes[cluster] >= min_cluster_size else -1) for cluster in clusters]

    clustered = pd.DataFrame(
        {
            "profile": zero_profiles + non_zero_profiles,
            "cluster": (len(zero_profiles) * [-2]) + list(clusters),
            "vector": zero_vectors + non_zero_vectors,
        }
    )

    cluster_names = find_cluster_names(clustered, profile_embedder.embedder)

    clustered["cluster"] = clustered["cluster"].map(cluster_names)
    return clustered, cluster_score


def compute_silhouette_score(clusters: typing.List[int], vectors: typing.List[NDArray]) -> float:
    """
    Computes the silhouette cluster Score for a given clustering.
    It filters the -1 cluster.

    :param clusters: Cluster numbers
    :param vectors: corresponding vectors to clusters
    :return: the computed score
    """
    silhouette_vectors = []
    silhouette_clusters = []
    for cluster, vector in zip(clusters, vectors):
        if cluster != -1:
            silhouette_clusters += [cluster]
            silhouette_vectors += [vector]
    cluster_score: float = silhouette_score(silhouette_vectors, silhouette_clusters)
    return cluster_score
