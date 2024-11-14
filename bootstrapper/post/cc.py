import numba
import numpy as np
from numba import jit


@jit(nopython=True)
def compute_connected_component_segmentation(hard_aff: np.ndarray) -> np.ndarray:
    """
    Compute connected components from affinities.

    Args:
        hard_aff: The (thresholded, boolean) short range affinities. Shape: (3, x, y, z).

    Returns:
        The segmentation. Shape: (x, y, z).
    """
    visited = np.zeros(tuple(hard_aff.shape[1:]), dtype=numba.boolean)
    seg = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.uint32)
    cur_id = 1
    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if hard_aff[:, i, j, k].any() and not visited[i, j, k]:  # If foreground
                    cur_to_visit = [(i, j, k)]
                    visited[i, j, k] = True
                    while cur_to_visit:
                        x, y, z = cur_to_visit.pop()
                        seg[x, y, z] = cur_id

                        # Check all neighbors
                        if (
                            x + 1 < visited.shape[0]
                            and hard_aff[0, x, y, z]
                            and not visited[x + 1, y, z]
                        ):
                            cur_to_visit.append((x + 1, y, z))
                            visited[x + 1, y, z] = True
                        if (
                            y + 1 < visited.shape[1]
                            and hard_aff[1, x, y, z]
                            and not visited[x, y + 1, z]
                        ):
                            cur_to_visit.append((x, y + 1, z))
                            visited[x, y + 1, z] = True
                        if (
                            z + 1 < visited.shape[2]
                            and hard_aff[2, x, y, z]
                            and not visited[x, y, z + 1]
                        ):
                            cur_to_visit.append((x, y, z + 1))
                            visited[x, y, z + 1] = True
                        if (
                            x - 1 >= 0
                            and hard_aff[0, x - 1, y, z]
                            and not visited[x - 1, y, z]
                        ):
                            cur_to_visit.append((x - 1, y, z))
                            visited[x - 1, y, z] = True
                        if (
                            y - 1 >= 0
                            and hard_aff[1, x, y - 1, z]
                            and not visited[x, y - 1, z]
                        ):
                            cur_to_visit.append((x, y - 1, z))
                            visited[x, y - 1, z] = True
                        if (
                            z - 1 >= 0
                            and hard_aff[2, x, y, z - 1]
                            and not visited[x, y, z - 1]
                        ):
                            cur_to_visit.append((x, y, z - 1))
                            visited[x, y, z - 1] = True
                    cur_id += 1
    return seg
