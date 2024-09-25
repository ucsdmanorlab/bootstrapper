import click
import logging

from bootstrapper.post.watershed import ws
# from bootstrapper.post.mws import mws
# from bootstrapper.post.threshold import threshold

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def segment():
    """
    Run post-processing on predictions to generate segmentations.

    Can be run blockwise or on the entire volume.
    """
    pass

segment.add_command(ws)
# segment.add_command(mws)
# segment.add_command(threshold)