from funlib.geometry import Roi, Coordinate
import numpy as np


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    is_3d = int(output_size[0] / voxel_size[0]) != 1

    if isinstance(sigma, int):
        sigma = Coordinate([sigma] * 3)

    method_padding = Coordinate(
        (
            sigma[0] * 3 * int(is_3d),
            sigma[1] * 3,
            sigma[2] * 3,
        )
    )

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = Roi(
        (
            Coordinate([i / 2 for i in [output_size[0] * int(is_3d), diag, diag]])
            + method_padding
        ),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()
