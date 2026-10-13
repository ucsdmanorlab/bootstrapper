"""Graph constraints and final ids for user-seeded mutex watershed. The db holds seed_id
as text, so ids above 2^31 fit a PostgreSQL column; it is cast to int where read."""
from itertools import combinations

import networkx as nx
import numpy as np

# score of a seed constraint edge: far above any real edge score |w * aff + b|
SEED_LINK = 1e6


def link_seeds(db, roi):
    """Chain the fragments of each seed id with must-link edges (+SEED_LINK) and
    join one fragment of every two seed ids with a mutex edge (-SEED_LINK), as
    edge attribute seed_link."""
    rag = db.open("r+")
    seeded = [n for n in rag.read_nodes(roi, read_attrs=["position", "seed_id"]) if int(n["seed_id"])]
    groups = {}
    for node in sorted(seeded, key=lambda n: n["id"]):
        groups.setdefault(int(node["seed_id"]), []).append(node["id"])
    links = [(u, v, SEED_LINK) for ids in groups.values() for u, v in zip(ids, ids[1:])]
    links += [(a[0], b[0], -SEED_LINK) for a, b in combinations(groups.values(), 2)]

    u_name, v_name = rag.endpoint_names
    existing = {(e[u_name], e[v_name]) for e in rag.read_edges(roi, nodes=seeded, read_attrs=[])}
    new, old = nx.Graph(), nx.Graph()
    for graph in (new, old):
        graph.add_nodes_from((n["id"], n) for n in seeded)
    for u, v, score in links:
        graph = old if (min(u, v), max(u, v)) in existing else new
        graph.add_edge(u, v, seed_link=score)
    rag.write_edges(new.nodes, new.edges)
    rag.update_edges(old.nodes, old.edges, attributes=["seed_link"])
    return len(groups), len(links)


def seed_range(store, num_workers):
    """The smallest and largest id in a seeds array, read chunk by chunk."""
    import dask
    from funlib.persistence import open_ds

    data = open_ds(store).data
    return tuple(int(v) for v in dask.compute(data.min(), data.max(), num_workers=num_workers))


def seed_lut(db, lut, roi, max_seed):
    """Rewrite the LUT over every fragment: the segment holding seed k becomes k,
    and the other segments count up from max_seed + 1."""
    nodes = db.open("r").read_nodes(roi, read_attrs=["seed_id"])
    frags, segs = lut.load()
    seg_of = dict(zip(frags.tolist(), segs.tolist()))
    segments = [seg_of.get(n["id"], n["id"]) for n in nodes]

    new_id = {}
    for node, seg in zip(nodes, segments):
        seed = int(node["seed_id"])
        if seed and new_id.setdefault(seg, seed) != seed:
            raise RuntimeError(f"seeds {new_id[seg]} and {seed} share segment {seg}")
    next_id = max_seed + 1
    for seg in segments:
        if seg not in new_id:
            new_id[seg] = next_id
            next_id += 1
    lut.save(
        np.array([[n["id"] for n in nodes], [new_id[s] for s in segments]], dtype=np.uint64),
        edges=np.load(lut.path)["edges"],
    )
