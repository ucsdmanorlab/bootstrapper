import os
import neuroglancer
import numpy as np
import networkx as nx
import operator
import sys
import itertools
from pathlib import Path

import zarr
from funlib.show.neuroglancer import add_layer
from funlib.persistence import open_ds
from funlib.persistence.graphs import SQLiteGraphDataBase

neuroglancer.set_server_bind_address('localhost',bind_port=3336)


f = sys.argv[1]

skels = [os.path.join(os.path.dirname(f),"oblique_skels.graphml"),]
print(skels)

f = zarr.open(f,"r")
#raw = f["raw"]
labels = f["labels/s0"]

vs = labels.attrs["resolution"] # voxel_size

def to_ng_coords(coords):
    return np.array([x/y for x,y in zip(coords,vs)]).astype(np.float32) + 0.5

def convert_edges_to_indices(edges, nodes):
    node_to_index = {node_id: index for index, node_id in enumerate(nodes)}
    edges_with_indices = [(node_to_index[src], node_to_index[dst]) for src, dst in edges]
    return edges_with_indices

class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, skel_file, dimensions):
        super().__init__(dimensions)
#        skel_provider = SQLiteGraphDataBase(
#                Path(skel_file),
#                position_attributes=['position_z', 'position_y', 'position_x'],
#                mode="r",
#                node_attrs={'label_id':int})
#
#        self.skeletons = skel_provider.read_graph()
        self.skeletons = nx.read_graphml(skel_file)
        print(skel_file, len(self.skeletons.nodes))

    def get_skeleton(self, i):
        filtered_nodes = [n for n, attr in self.skeletons.nodes(data=True) if attr.get('id') == i] 
        subgraph = self.skeletons.subgraph(filtered_nodes)
        
        vertex_positions = []
        
        for u, data in subgraph.nodes(data=True):
            pos = [data['position_z'], data['position_y'], data['position_x']]
            pos = to_ng_coords(pos)
            vertex_positions.append(pos)


        edges = convert_edges_to_indices(subgraph.edges(),subgraph.nodes())

        vertex_position = np.array(vertex_positions)
        edges = np.array(edges)

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            #vertex_attributes=dict(affinity=gen.rand(2), affinity2=gen.rand(2)),
        )


dims = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=vs)
viewer = neuroglancer.Viewer()

uniques = labels[:].copy()
uniques = uniques[uniques > 0]
uniques = np.unique(uniques)
uniques = list(uniques)

with viewer.txn() as s: 
#    s.layers.append(
#            name="raw",
#            layer=neuroglancer.ImageLayer(
#                source=neuroglancer.LocalVolume(
#                        data=raw,
#                        dimensions=dims,
#                        voxel_offset=[x/y for x,y in zip(raw.attrs["offset"],vs)],
#                    ),
#            )
#    )
    s.layers.append(
            name="labels",
            layer=neuroglancer.SegmentationLayer(
                source=neuroglancer.LocalVolume(
                        data=labels[:],
                        dimensions=dims,
                        voxel_offset=[x/y for x,y in zip(labels.attrs["offset"],vs)],
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

print(viewer)
