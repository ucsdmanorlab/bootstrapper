import os
import sys
import numpy as np
import networkx as nx
import neuroglancer
import zarr


def to_ng_coords(coords, voxel_size):
    return np.array([x / y for x, y in zip(coords, voxel_size)]).astype(np.float32) + 0.5

def convert_edges_to_indices(edges, nodes):
    node_to_index = {node_id: index for index, node_id in enumerate(nodes)}
    return [(node_to_index[src], node_to_index[dst]) for src, dst in edges]

class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, skel_file, dimensions):
        super().__init__(dimensions)
        self.skeletons = nx.read_graphml(skel_file)
        print(f"Loaded {len(self.skeletons.nodes)} nodes from {skel_file}")

    def get_skeleton(self, i):
        filtered_nodes = [n for n, attr in self.skeletons.nodes(data=True) if attr.get('id') == i]
        subgraph = self.skeletons.subgraph(filtered_nodes)
        
        vertex_positions = [to_ng_coords([data['position_z'], data['position_y'], data['position_x']], self.dimensions.scales)
                            for _, data in subgraph.nodes(data=True)]
        
        edges = convert_edges_to_indices(subgraph.edges(), subgraph.nodes())
        
        return neuroglancer.skeleton.Skeleton(
            vertex_positions=np.array(vertex_positions),
            edges=np.array(edges),
        )

def run_viewer(labels_store, skels):
    labels = zarr.open(labels_store, "r")
    voxel_size = labels.attrs["voxel_size"]
    
    dims = neuroglancer.CoordinateSpace(names=labels.attrs["axis_names"], units=labels.attrs["units"], scales=voxel_size)
    viewer = neuroglancer.Viewer()
    
    uniques = np.unique(labels[:] > 0)
    
    with viewer.txn() as s:
        s.layers.append(
            name="labels",
            layer=neuroglancer.SegmentationLayer(
                source=neuroglancer.LocalVolume(
                    data=labels[:],
                    dimensions=dims,
                    voxel_offset=[x/y for x,y in zip(labels.attrs["offset"], voxel_size)],
                ),
            )
        )
        
        for skel in skels:
            name = os.path.basename(skel)
            s.layers.append(
                name=name,
                layer=neuroglancer.SegmentationLayer(
                    source=SkeletonSource(skel, dims),
                    linked_segmentation_group="labels",
                    segments=uniques,
                )
            )
            s.layers[name].skeleton_rendering.line_width3d = 3
    
    return viewer

def main():
    neuroglancer.set_server_bind_address('0.0.0.0')
    
    if len(sys.argv) < 3:
        print("Usage: python -i iew_ng_skeletons.py <labels_store> <skel_file1> [<skel_file2> ...]")
        sys.exit(1)
    
    labels_store = sys.argv[1]
    skel_files = sys.argv[2:]
    
    viewer = run_viewer(labels_store, skel_files)
    print(viewer)

if __name__ == "__main__":
    main()