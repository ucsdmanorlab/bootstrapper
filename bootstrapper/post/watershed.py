import click
import logging
import yaml

from .blockwise.hglom.frags import frags, extract_fragments
from .blockwise.hglom.agglom import agglom, agglomerate
from .blockwise.hglom.luts import luts, find_segments
from .blockwise.hglom.extract import extract, extract_segmentations


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class OrderedGroup(click.Group):
    def list_commands(self, ctx):
        # Return the commands in the desired order
        return [
            "frags",
            "agglom",
            "luts",
            "extract",
            "pipeline",
        ]


@click.group(cls=OrderedGroup)
def ws():
    """
    Hierarchical region agglomeration using waterz.
    """
    pass

ws.add_command(frags)
ws.add_command(agglom)
ws.add_command(luts)
ws.add_command(extract)

@ws.command()
@click.argument("config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def pipeline(config_file):
    """
    Run the watershed pipeline.

    frags -> agglom -> luts -> extract

    """
    # Load config file
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["waterz"]
    db_config = yaml_config["db"]

    extract_fragments(config, db_config)
    agglomerate(config, db_config)
    find_segments(config, db_config)
    extract_segmentations(config | db_config)
