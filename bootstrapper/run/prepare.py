import click
import os
import yaml
import logging

from .configs import make_configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--base-dir",
    "-b",
    required=True,
    type=click.Path(file_okay=False),
    help="Base directory for the bootstrapping runs",
    prompt="Enter the base directory path",
    default=".",
    show_default=True,
)
def prepare(base_dir):
    """
    Prepare and configure a multi-round bootstrapping pipeline for image segmentation.

    This command sets up a pipeline for training, prediction, post-processing,
    evaluation, and filtering segmentations across multiple rounds. It handles:

    - Volume preparation: Process raw and label data (tif/zarr)

    - Configuration generation for each round:

        - Training: Set model and parameters

        - Prediction: Configure for each volume

        - Segmentation: Set up RAG, blockwise for each volume

        - Evaluation: Configure lsd error metrics and ground truth comparisons

        - Filtering: Set up pseudo ground truth generation

    The process is iterative, with each round building upon previous results,
    allowing for refinement of the segmentations over time.

    Usage:
        Run the `prepare` function to start the bootstrapping setup process.
        Follow the prompts to customize the pipeline for your specific needs.

    Returns:
        None. Generates configuration files in the specified directories.
    """

    os.makedirs(base_dir, exist_ok=True)
    base_dir = os.path.abspath(base_dir)
    logger.info(f"Base directory: {base_dir}")

    configs = make_configs(base_dir)
