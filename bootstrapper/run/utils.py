import click
from bootstrapper.utils.bbox import bbox
from bootstrapper.utils.clahe import clahe
from bootstrapper.utils.convert import convert
from bootstrapper.utils.mask import mask
from bootstrapper.utils.outlier_filter import outlier_filter
from bootstrapper.utils.scale_pyramid import scale_pyramid
from bootstrapper.utils.size_filter import size_filter


@click.group()
def utils():
    """Utility functions for volumes and segmentations"""
    pass


utils.add_command(bbox)
utils.add_command(clahe)
utils.add_command(convert)
utils.add_command(mask)
utils.add_command(outlier_filter)
utils.add_command(scale_pyramid)
utils.add_command(size_filter)
