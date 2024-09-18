import gunpowder as gp
from skimage.measure import label


class Renumber(gp.BatchFilter):
    """Find connected components of the same value, and replace each component
    with a new label.

    Args:

        labels (:class:`ArrayKey`):

            The label array to modify.
    """

    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        components = batch.arrays[self.labels].data
        dtype = components.dtype

        components = label(components, connectivity=1)
        batch.arrays[self.labels].data = components.astype(dtype)
