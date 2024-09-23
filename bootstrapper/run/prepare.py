import click
import os
import yaml
import logging

from .configs import make_configs

# Set up logging
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
    show_default=True
)
def prepare(base_dir):
    """Prepare the volumes and configuration files for bootstrapping."""

    os.makedirs(base_dir, exist_ok=True)
    base_dir = os.path.abspath(base_dir)
    logger.info(f"Base directory: {base_dir}")

    configs = make_configs(base_dir)