"""bs view: what the steps of a round wrote, a training snapshot, or array paths, in
neuroglancer. layer_type gives every array its layer from its name, dtype and channels."""
import glob
import json
import os

import click
import neuroglancer
import numpy as np
import zarr
from funlib.persistence import open_ds

from .config import load_volumes, matches, numbered, paths, runs
from .configs import output_level

RGB = """void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2))));
}"""
LSD_CHANNELS = {"2d_lsds": "offset y x, variance y", "3d_lsds": "offset z y x"}


@click.command()
@click.option(
    "--snapshot",
    "-s",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to the Zarr container of a snapshot",
)
@click.option(
    "--bind",
    "-b",
    type=str,
    default="0.0.0.0",
    help="Address to serve the viewer on",
)
@click.argument("path", nargs=-1)
def view(snapshot, bind, path):
    """View a round (bs view round_01), one step of it (bs view round_01 04), a training
    snapshot (-s), or array paths. Prints each layer, then the URL."""
    if not snapshot and not path:
        raise click.UsageError("give a round, array paths, or --snapshot")
    neuroglancer.set_server_bind_address(bind)
    viewer = neuroglancer.Viewer()
    if snapshot:
        layers = snapshot_layers(snapshot.rstrip("/"))
    elif _is_round(path[0]):
        layers = round_layers(path[0], path[1:])
    else:
        names = _unique([os.path.basename(p.rstrip("/")) for p in path])
        layers = [(n, a, *_array(n, a), True) for n, a in zip(names, map(_finest, path))]

    width = max(len(name) for name, *_ in layers) if layers else 0
    with viewer.txn() as s:
        for name, array_path, kind, source, visible in layers:
            click.echo(f"{name:<{width}}  {array_path}  {kind}")
            s.layers.append(name=name, layer=_layer(kind, source), visible=visible)
        if snapshot:
            s.layout = "yz"
        elif layers:
            s.dimensions = _xyz_first(layers[0][3].dimensions)
    click.echo(viewer)
    click.pause("Press any key to exit...")


def create_coordinate_space(voxel_size, is_2d):
    names = ["c^", "z", "y", "x"] if not is_2d else ["b", "c^", "y", "x"]
    scales = (
        [
            1,
        ]
        + voxel_size
        if not is_2d
        else voxel_size[-2:] + voxel_size[-2:]
    )
    return neuroglancer.CoordinateSpace(names=names, units="nm", scales=scales)


def process_dataset(f, ds, is_2d):
    data = f[ds][:]
    try:
        vs = f[ds].attrs["voxel_size"]
    except:
        vs = f[ds].attrs["resolution"]
    offset = f[ds].attrs["offset"]

    if is_2d:
        if ds != "raw" and len(data.shape) == 5:
            data = np.squeeze(data, axis=-3)
            offset = offset[1:]
            vs = vs[1:]
        elif ds == "raw" and len(data.shape) == 4 and len(vs) == 3:
            vs = vs[1:]

    if is_2d:
        offset = [0, 0] + [int(i / j) for i, j in zip(offset, vs)]
    else:
        offset = [
            0,
        ] + [int(i / j) for i, j in zip(offset, vs)]

    return data, vs, offset


def snapshot_layers(zarr_path):
    f = zarr.open(zarr_path)
    datasets = list(f.array_keys())

    # Determine if the data is 3D based on the first dataset
    try:
        raw_shape = f["raw"].shape
    except KeyError:
        raw_shape = f[datasets[0]].shape
    shape = f[datasets[0]].shape
    if len(shape) == 5:
        is_2d = (shape[-3] == 1) and (len(raw_shape) == 4)
    elif len(shape) == 4:
        if raw_shape[0] == 1:
            is_2d = False
        else:
            is_2d = raw_shape != shape
        #is_2d = (len(raw_shape) == 4) and raw_shape[0] != 1
    else:
        is_2d = False

    try:
        vs = f[datasets[0]].attrs["voxel_size"]
    except:
        vs = f[datasets[0]].attrs["resolution"]

    dims = create_coordinate_space(vs, is_2d)
    layers = []
    for ds in datasets:
        data, _, offset = process_dataset(f, ds, is_2d)
        kind = layer_type(ds, data.dtype, data.shape[dims.names.index("c^")])
        source = neuroglancer.LocalVolume(data=data, voxel_offset=offset, dimensions=dims,
                                          volume_type="segmentation" if kind == "segmentation" else "image")
        layers.append((ds, f"{zarr_path}/{ds}", kind, source, True))
    return layers


def layer_type(name, dtype, channels):
    """rgb image (channels 0, 1, 2) for affinities, LSDs and the sections of a 2D batch's raw,
    segmentation for ids and masks, else a grey image."""
    if ("affs" in name or "lsds" in name or name == "raw") and channels >= 3:
        return "rgb image"
    if np.issubdtype(dtype, np.integer) and ("label" in name or "mask" in name or np.dtype(dtype).itemsize >= 4):
        return "segmentation"
    return "image"


def _xyz_first(dims):
    """The dimensions with x, y, z first, so the xy panel shows a section the right way round."""
    order = [n for n in ("x", "y", "z") if n in dims.names] + [n for n in dims.names if n not in "xyz" and "^" not in n]
    i = [dims.names.index(n) for n in order]
    return neuroglancer.CoordinateSpace(names=order, units=[dims.units[k] for k in i], scales=[dims.scales[k] for k in i])


def _layer(kind, source):
    if kind == "segmentation":
        return neuroglancer.SegmentationLayer(source=source)
    return neuroglancer.ImageLayer(source=source, **({"shader": RGB} if kind == "rgb image" else {}))


def _array(name, path):
    """The layer type and the source of an array on disk."""
    a = open_ds(path)
    spatial = iter(zip(a.voxel_size, a.units, a.offset / a.voxel_size))
    scales, units, offset = zip(*(next(spatial) if "^" not in n else (1, "", 0) for n in a.axis_names))
    channels = next((size for n, size in zip(a.axis_names, a.shape) if "^" in n), 1)
    kind = layer_type(name, a.dtype, channels)
    source = neuroglancer.LocalVolume(
        zarr.open(path, mode="r"),
        dimensions=neuroglancer.CoordinateSpace(names=a.axis_names, units=units, scales=scales),
        voxel_offset=offset,
        volume_type="segmentation" if kind == "segmentation" else "image",
    )
    return kind, source


def _is_array(path):
    try:
        return isinstance(zarr.open(path, mode="r"), zarr.Array)
    except ValueError:
        return False


def _finest(path):
    """A scale-pyramid group opens at its s0."""
    s0 = os.path.join(path, "s0")
    return s0 if not _is_array(path) and _is_array(s0) else path


def _is_round(path):
    return path.endswith(".toml") or bool(glob.glob(os.path.join(glob.escape(path), "[0-9]*.toml")))


def _unique(names):
    """The names, with a count after the second and later of a name."""
    seen = {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        yield n if seen[n] == 1 else f"{n} {seen[n]}"


def round_layers(path, chosen):
    """For each volume: the raw, then what the chosen steps wrote (all steps: also the
    volume's own arrays). An array belongs to the last step that writes it, so a refine
    output beside a segmentation is the refine step's. The raw and the newest array of the
    first volume start visible."""
    todo = numbered(path)
    numbers = [n for n, f, s in todo if not chosen or any(matches(c, n, f, s) for c in chosen)]
    if not numbers:
        raise click.ClickException(f"{' '.join(chosen)}: no such step; steps are " + ", ".join(f"{n} {s.label}" for n, _, s in todo))
    owner = {}  # array path: (step number, volume, layer name), in step order
    for n, _, s in todo:
        for vol, name, p in _written(s):
            owner.pop(p, None)
            owner[p] = (n, vol, name)
    volumes = load_volumes(todo[0][1])
    shown = {} if chosen else {v: [] for v in volumes}
    for p, (n, vol, name) in owner.items():
        if n in numbers and _is_array(p):
            shown.setdefault(vol, []).append((f"{n} {name}", p))

    layers = []
    for i, (vol, rows) in enumerate(shown.items()):
        keys = volumes[vol].keys if vol in volumes else {}
        own = [(k, p) for k, p in keys.items() if k != "raw" and isinstance(p, str) and "/" in p and not chosen and _is_array(p)]
        rows = ([("raw", keys["raw"])] if keys else []) + own + rows
        prefix = f"{vol} " if vol and len(shown) > 1 else ""
        names = _unique([prefix + name for name, _ in rows])
        for j, (label, (name, p)) in enumerate(zip(names, rows)):
            visible = i == 0 and (name == "raw" or j == len(rows) - 1)
            layers.append((label, p, *_array(name, p), visible))
    return layers


def _written(step):
    """(volume, layer name, path) of each array the step writes."""
    for vol, c in runs(step):
        if step.kind == "train":
            yield from ((s.get("volume"), "training_mask", s["mask"]) for s in c.get("samples", []) if "mask" in s)
        elif step.kind == "predict":
            with open(os.path.join(c["setup_dir"], "net_config.json")) as f:
                outputs = json.load(f)["outputs"]
            level = output_level(c["checkpoint"], c.get("chain_str", ""))
            for out, net in outputs.items():
                channels = _channels(out, net)
                yield vol, out + (f" ({channels})" if channels else ""), os.path.join(c["output_datasets_prefix"], level, out)
        elif step.kind == "segment":
            for key, name in (("fragments_dataset", "fragments"), ("seg_dataset_prefix", "segmentation")):
                for p in paths(c, key):
                    found = [p] if _is_array(p) else sorted(glob.glob(os.path.join(glob.escape(p), "*")))
                    yield from ((vol, name, a) for a in found)
        elif step.kind == "refine":
            yield from ((vol, p[len(c["in_array"]) + 1:], p) for p in sorted(glob.glob(glob.escape(c["in_array"]) + "_*")))
        elif step.kind == "evaluate":
            segs = [os.path.join(glob.escape(d), "*") for k in ("out_dir", "seg_datasets_prefix") for d in paths(c, k)]
            segs += [glob.escape(d) for d in paths(c, "seg_datasets")]
            yield from ((vol, os.path.basename(p), p) for seg in segs for p in sorted(glob.glob(seg + "__vs__*/error_*")))


def _channels(output, net):
    """The first three channels, which the rgb shader shows: affinity offsets as axes, such
    as z y x or y x y9, or LSD components."""
    if "neighborhood" not in net:
        return LSD_CHANNELS.get(output)
    return " ".join("zyx"[-len(offset):][i] + (str(abs(d)) if abs(d) > 1 else "")
                    for offset in net["neighborhood"][:3] for i, d in enumerate(offset) if d)
