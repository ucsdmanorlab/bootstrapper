import click
from bootstrapper.data.bbox import bbox
from bootstrapper.data.clahe import clahe
from bootstrapper.data.convert import convert
from bootstrapper.data.mask import mask
from bootstrapper.data.scale_pyramid import scale_pyramid


@click.group()
def utils():
    """Utility functions for volumes and segmentations"""
    pass


utils.add_command(bbox)
utils.add_command(clahe)
utils.add_command(convert)
utils.add_command(mask)
utils.add_command(scale_pyramid)