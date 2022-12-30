"""
This modules provides commands for profile clustering
"""
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

import click
import pandas as pd
import plotly.io as pio
from tqdm import tqdm

from profile_extraction.profile_clustering.cluster import (
    EMPTY_PROFILE,
    cluster_profiles,
    tsne_plot,
)
from profile_extraction.profile_clustering.embedding import (
    ProfileEmbedder,
    get_profile_embedder,
)
from profile_extraction.profile_creation.profile import Profile, ProfileCollection

LOG_LEVELS = ["CRITICAL", "ERROR", "WARN", "INFO", "DEBUG"]
EMBEDDING_TYPES = [
    ProfileEmbedder.WORD,
    ProfileEmbedder.FLAIR,
    ProfileEmbedder.TRANSFORMER,
    ProfileEmbedder.MODEL,
    ProfileEmbedder.LOWERCASED_FASTTEXT,
]

log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, resolve_path=True),
    required=True,
    help="Path to a SequenceTagger model providing NER for profile clustering",
)
@click.option(
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False, resolve_path=True),
    required=True,
    help="directory to put the clustering results",
)
@click.option(
    "--visualizations-path",
    type=click.Path(dir_okay=True, file_okay=False, resolve_path=True, exists=True),
    required=True,
    help="directory containing visualized profiles",
)
@click.option(
    "--profile-path",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, resolve_path=True),
    required=True,
    help="Path to a JSON containing generated profiles",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.1,
    help="Max cosine distance between two clusters to merge. Lower values are equal to stricter clustering.",
)
@click.option("--log-level", type=click.Choice(LOG_LEVELS), default="INFO", help="log level to use defaults to INFO")
@click.option(
    "--linkage",
    type=click.Choice(["complete", "single", "average"]),
    default="average",
    help="linkage method to use for clustering",
)
@click.option("--tf-idf/--no-tf-idf", default=True, help="Whether tf-idf should be used for product weights")
@click.option(
    "--contextualized/--not-contextualized",
    default=False,
    help="Whether to use contextualized Transformer Embeddings. "
    "Only Works with a Transformer model and ignores tf-idf option",
)
@click.option("--embeddings", default=None)
@click.option("--embeddings-type", type=click.Choice(EMBEDDING_TYPES), default=ProfileEmbedder.MODEL)
def cmd_cluster_profiles(  # pylint: disable=too-many-locals
    model_path: str,
    embeddings: Optional[str],
    embeddings_type: str,
    output_dir,
    visualizations_path: Union[str, Path],
    profile_path: str,
    threshold: float,
    log_level: str,
    linkage: str,
    tf_idf: bool,
    contextualized: bool,
):
    """
    Clusters profiles using PROD token embeddings
    :param contextualized: Whether to use contextualized Transformer Embeddings
    :param tf_idf: Whether to use tf-idf weighting of products
    :param linkage: linkage type to use
    :param visualizations_path: Path to write the visualizations to
    :param log_level: log level to use defaults to INFO
    :param threshold: Max cosine distance between two clusters to merge. Lower values are equal to stricter clustering
    :param model_path: Model to use for clustering
    :param output_dir: Directory to write results to
    :param profile_path: File containing profiles
    """
    logging.basicConfig(level=log_level)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizations_path = Path(visualizations_path)

    log.info("Reading Profiles...")
    profiles = ProfileCollection.parse_file(profile_path)

    log.info("Clustering Profiles...")
    embeddings = model_path if not embeddings else embeddings
    profile_embedder: ProfileEmbedder = get_profile_embedder(
        model_path, embeddings, tf_idf, contextualized, embeddings_type
    )
    clustered, score = cluster_profiles(
        profiles,
        profile_embedder,
        distance_threshold=threshold,
        linkage=linkage,
    )

    write_results(clustered, visualizations_path, output_dir, score)


def write_results(clustered: pd.DataFrame, visualizations_path: Path, output_dir: Path, score: float):
    """
    Writes clustering results to the filesystem
    :param clustered: DataFrame containing the clustered Profiles
    :param visualizations_path: Path to get profile visualizaitons from
    :param output_dir: Directory to write the restults to
    """
    cluster_fig = tsne_plot(clustered)
    pio.kaleido.scope.mathjax = None
    cluster_fig.write_html(output_dir / f"clusters_{score:.2f}.html")
    cluster_fig.write_image(output_dir / f"clusters_{score:.2f}.pdf")

    profiles_to_write = clustered.loc[clustered["cluster"] != EMPTY_PROFILE, :]
    for _, row in tqdm(
        profiles_to_write.iterrows(),
        total=len(profiles_to_write),
        desc="Creating links to profiles",
    ):
        profile: Profile = row["profile"]
        user_id = profile.user.id
        cluster_folder = str(row["cluster"]).replace("/", "_").strip()
        cluster_path = output_dir / cluster_folder[: min(100, len(cluster_folder))]
        cluster_path.mkdir(exist_ok=True)
        shutil.copy(visualizations_path / f"{user_id}.html", cluster_path / f"{user_id}.html")


if __name__ == "__main__":
    cmd_cluster_profiles()  # pylint: disable=no-value-for-parameter
