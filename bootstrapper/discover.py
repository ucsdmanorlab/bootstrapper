"""The arrays of a zarr container, from their metadata alone."""
import glob
import os
import re
from dataclasses import dataclass

import zarr


@dataclass(frozen=True)
class Array:
    path: str
    name: str  # the path in its container
    shape: tuple
    dtype: str
    voxel_size: tuple
    offset: tuple

    @property
    def role(self):
        """A guess the answer confirms: a mask by name, labels by integer type, else an image."""
        if "mask" in self.name.lower():
            return "mask"
        return "labels" if self.dtype in ("int32", "int64", "uint32", "uint64") else "image"

    @property
    def roi(self):
        return [(o, o + s * v) for o, s, v in zip(self.offset, self.shape[-len(self.voxel_size):], self.voxel_size)]


def container(path):
    """The zarr container a path is in, up to its .zarr; a path without one is in its folder."""
    return path.rsplit(".zarr", 1)[0] + ".zarr" if ".zarr" in path else os.path.dirname(path)


def is_array(path):
    return os.path.exists(os.path.join(path, ".zarray"))


def read(path, root=""):
    a = zarr.open_array(path, mode="r")
    voxel_size = tuple(a.attrs.get("voxel_size", (1,) * a.ndim))
    name = os.path.relpath(path, root) if root and path.startswith(root + os.sep) else path
    return Array(path, name, a.shape, str(a.dtype), voxel_size,
                 tuple(a.attrs.get("offset", (0,) * len(voxel_size))))


def arrays(group):
    """Every array under a zarr group, in path order."""
    found = []
    for root, dirs, files in os.walk(group):
        if ".zarray" in files:
            found.append(read(root, group))
        dirs[:] = [] if ".zarray" in files else sorted(d for d in dirs if not d.startswith("."))
    return found


def scales(array):
    """The scales of an array named like s0: the arrays named like it beside it."""
    if not re.fullmatch(r"s\d+", os.path.basename(array.path)):
        return [array]
    return [read(p) for p in sorted(glob.glob(os.path.join(os.path.dirname(array.path), "s[0-9]*"))) if is_array(p)]


def candidates(raw, arrays):
    """The labels and masks that fit the raw: its voxel size, and a region that overlaps its own."""
    fit = [a for a in arrays if a.path != raw.path and a.voxel_size == raw.voxel_size
           and all(lo < raw_hi and raw_lo < hi for (lo, hi), (raw_lo, raw_hi) in zip(a.roi, raw.roi))]
    return {role: [a for a in fit if a.role == role] for role in ("labels", "mask")}
