"""
This module coordinates profile visualization as HTML files
"""
from pathlib import Path

import click
from tqdm import tqdm

from profile_extraction.profile_creation.profile import ProfileCollection
from profile_extraction.profile_visualization.profilevis import (
    create_profile_visualization,
)


@click.command()
@click.option("--input-json", type=click.Path(dir_okay=False, file_okay=True, readable=True), required=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    required=True,
)
def cmd_profiles_to_html(input_json, output_dir):
    """
    Creates HTML profile visualizations form a given JSON file
    :param input_json: Json containing a ProfileCollection
    :param output_dir: Diretory to generate the HTMLs into (one file per user)
    :param model_path: Path to the used NER model (for visualization of Named entities)
    """

    input_json = Path(input_json)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    data = ProfileCollection.parse_file(input_json)

    for profile in tqdm(data.profiles, desc="Creating visualizations"):
        create_profile_visualization(output_dir, profile)


if __name__ == "__main__":
    cmd_profiles_to_html()  # pylint: disable=no-value-for-parameter
