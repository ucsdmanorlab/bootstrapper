"""User seeds ([segment] seeds_dataset) keep their ids through blockwise mutex watershed:
two seeds part touching objects, one seed holds an object over many blocks, and unseeded
objects get ids above every seed id. Also the segment ROI checks of mutex watershed."""

import getpass
import glob
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import toml
import zarr
from funlib.persistence import prepare_ds

from bootstrapper.post.naming import dump_params, inputs_differ, mws_segmentation_name
from bootstrapper.post.watershed_mutex import mutex_watershed_segmentation
from bootstrapper.segment import DEFAULTS, get_seg_config, run_segmentation

BS = os.path.join(os.path.dirname(sys.executable), "bs")
VS = (50, 4, 4)
SHAPE = (32, 256, 256)
NB = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-2, 0, 0],
    [0, -9, 0],
    [0, 0, -9],
    [-3, 0, 0],
    [0, -27, 0],
    [0, 0, -27],
]
PARAMS = {
    "aff_neighborhood": NB,
    "bias": [-0.5] * 3 + [-0.8] * 6,
    "strides": [[1, 1, 1]] * 3 + [[2, 9, 9]] * 3 + [[3, 27, 27]] * 3,
    "noise_eps": 0.001,
    "randomized_strides": True,
    "filter_fragments": 0.1,
    "remove_debris": 64,
}
BIG = 2**40

# object: gt id, z, y, x voxel ranges; blocks are 16 x 64 x 64
BLOBS = {
    "A": (1, (20, 30), (20, 44), (30, 170)),  # spans x blocks 0, 1, 2
    "B": (2, (2, 14), (90, 128), (70, 120)),  # touches C on the y=128 block face
    "C": (3, (2, 14), (128, 166), (70, 120)),
    "D": (4, (2, 14), (180, 250), (180, 250)),  # 4 blocks, seed in one
    "E": (5, (18, 30), (150, 250), (20, 80)),  # unseeded
    "F": (6, (18, 30), (70, 100), (200, 230)),  # touches G inside one block
    "G": (7, (18, 30), (100, 124), (200, 230)),
    "H": (8, (2, 14), (10, 60), (150, 240)),
}
# seed id: object, z, y, x voxel ranges
SEEDS = {
    104: ("A", (24, 26), (30, 34), (40, 160)),  # one seed over 3 blocks
    5: (
        "B",
        (6, 10),
        (100, 110),
        (85, 105),
    ),  # 18 voxels from the face: outside C's context
    13: ("C", (6, 10), (146, 156), (85, 105)),
    42: ("D", (6, 10), (200, 210), (200, 210)),
    7: ("F", (22, 26), (75, 80), (210, 215)),
    9: ("G", (22, 26), (115, 120), (210, 215)),
    BIG: (
        "H",
        (6, 10),
        (30, 40),
        (180, 200),
    ),  # above 2^31: a PostgreSQL INTEGER cannot hold it
}


def box(ranges):
    return tuple(slice(a, b) for a, b in ranges)


def write(path, data, offset=(0, 0, 0)):
    ds = prepare_ds(
        path,
        shape=data.shape,
        offset=offset,
        voxel_size=VS,
        axis_names=["c^"] * (data.ndim - 3) + ["z", "y", "x"],
        units=["nm"] * 3,
        chunk_shape=data.shape[:-3] + (16, 64, 64),
        dtype=data.dtype,
    )
    ds[ds.roi] = data


@pytest.fixture(scope="module")
def volume(tmp_path_factory):
    """uint8 affs with no boundary between B and C or between F and G: only seeds part them."""
    d = tmp_path_factory.mktemp("seeded")
    gt = np.zeros(SHAPE, np.uint64)
    for gid, *r in BLOBS.values():
        gt[box(r)] = gid
    merged = np.where(gt == 3, 2, np.where(gt == 7, 6, gt))
    affs = np.zeros((len(NB), *SHAPE), np.float32)
    for c, o in enumerate(NB):
        a = tuple(slice(-m, None) if m else slice(None) for m in o)
        b = tuple(slice(0, s + m) if m else slice(None) for m, s in zip(o, SHAPE))
        affs[c][a] = (merged[a] == merged[b]) & (merged[a] > 0)
    affs = np.clip(affs + np.random.default_rng(0).normal(0, 0.05, affs.shape), 0, 1)
    write(f"{d}/data.zarr/affs", np.round(affs * 255).astype(np.uint8))

    seeds = np.zeros(SHAPE, np.uint64)
    for sid, (_, *r) in SEEDS.items():
        seeds[box(r)] = sid
    # the seeds array covers the seeds' bounding box only, at an offset
    lo, hi = zip(*[(int(i.min()), int(i.max()) + 1) for i in np.nonzero(seeds)])
    write(
        f"{d}/data.zarr/seeds",
        seeds[box(zip(lo, hi))],
        offset=[a * v for a, v in zip(lo, VS)],
    )
    return SimpleNamespace(dir=d, gt=gt, seeds=seeds)


def segment(volume, out, db, seeded):
    """`bs segment -mws` over 32 blocks; returns the name and data of the new segmentation."""
    d = volume.dir
    step = {
        "method": "mws",
        "affs_dataset": f"{d}/data.zarr/affs",
        "fragments_dataset": f"{d}/{out}/fragments_mws",
        "seg_dataset_prefix": f"{d}/{out}/segmentations_mws",
        "lut_dir": f"{d}/{out}/luts_mws",
        "blockwise": True,
        "num_workers": 4,
        "block_shape": [16, 64, 64],
        "context": [4, 16, 16],
        "params": PARAMS,
        "db": db,
    }
    if seeded:
        step["seeds_dataset"] = f"{d}/data.zarr/seeds"
    config = d / f"{out}_{'seeded' if seeded else 'unseeded'}.toml"
    config.write_text(toml.dumps({"segment": step}))
    run = subprocess.run(
        [BS, "segment", str(config), "-mws"], cwd=d, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout + run.stderr
    [name] = [
        n
        for n in os.listdir(f"{d}/{out}/segmentations_mws")
        if ("seeded" in n.split("--")) == seeded
    ]
    return SimpleNamespace(
        name=name, data=zarr.open(f"{d}/{out}/segmentations_mws/{name}", "r")[:]
    )


@pytest.fixture(scope="module")
def unseeded(volume):
    return segment(
        volume, "out.zarr", {"db_file": f"{volume.dir}/unseeded.db"}, seeded=False
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("sqlite", marks=pytest.mark.smoke),
        pytest.param("postgresql", marks=[pytest.mark.smoke, pytest.mark.postgres]),
    ],
)
def seeded(request, volume):
    """The seeded run, its graph in SQLite or PostgreSQL (skipped when no login works)."""
    if request.param == "sqlite":
        yield segment(
            volume, "out.zarr", {"db_file": f"{volume.dir}/seeded.db"}, seeded=True
        )
        return
    psycopg2 = pytest.importorskip("psycopg2")
    login = {
        "host": os.environ.get("PGHOST", "localhost"),
        "user": os.environ.get("PGUSER", getpass.getuser()),
        "password": os.environ.get("PGPASSWORD", ""),
    }
    try:
        psycopg2.connect(dbname="postgres", connect_timeout=3, **login).close()
    except psycopg2.OperationalError as e:
        pytest.skip(f"cannot log in to PostgreSQL: {e}".splitlines()[0])
    name = f"bs_test_seeded_{os.getpid()}"
    db = {
        "db_host": login["host"],
        "db_user": login["user"],
        "db_password": login["password"],
        "db_name": name,
    }
    try:
        yield segment(volume, "pg.zarr", db, seeded=True)
    finally:
        con = psycopg2.connect(dbname="postgres", **login)
        con.autocommit = True
        con.cursor().execute(f"DROP DATABASE IF EXISTS {name}")
        con.close()


def ids_in(seg, volume, blob):
    gid, *r = BLOBS[blob]
    return set(np.unique(seg[box(r)][volume.gt[box(r)] == gid]).tolist())


@pytest.mark.parametrize(
    "blob,seed",
    [("B", 5), ("C", 13), ("A", 104), ("D", 42), ("H", BIG)],
    ids=[
        "touching_B",
        "touching_C",
        "three_blocks_A",
        "past_its_seed_D",
        "above_2^31_H",
    ],
)
def test_a_seeded_object_is_its_seed_id(volume, seeded, blob, seed):
    assert ids_in(seeded.data, volume, blob) == {seed}


def test_an_unseeded_object_gets_one_id_above_every_seed_id(volume, seeded):
    [only] = ids_in(seeded.data, volume, "E")
    assert only > BIG


def test_objects_touching_in_one_block_get_only_their_two_seed_ids(volume, seeded):
    assert ids_in(seeded.data, volume, "F") | ids_in(seeded.data, volume, "G") == {7, 9}


def test_seed_voxels_keep_their_ids(volume, seeded):
    at_seeds = volume.seeds > 0
    assert (seeded.data[at_seeds] == volume.seeds[at_seeds]).all()


def test_unseeded_ids_are_above_every_seed_id(seeded):
    seg = seeded.data
    assert (seg[(seg > 0) & ~np.isin(seg, list(SEEDS))] > BIG).all()


@pytest.mark.smoke
@pytest.mark.parametrize("x,y", [("B", "C"), ("F", "G")])
def test_without_seeds_the_touching_objects_merge(volume, unseeded, x, y):
    ids = ids_in(unseeded.data, volume, x)
    assert len(ids) == 1 and ids == ids_in(unseeded.data, volume, y)


def test_prepare_predicts_the_names_a_run_writes(seeded, unseeded):
    assert seeded.name == mws_segmentation_name({**PARAMS, "seeds_dataset": "seeds"})
    assert seeded.name.startswith("seeded--")
    assert unseeded.name == mws_segmentation_name(PARAMS)


def default_db(**extra):
    step = {
        "affs_dataset": "a",
        "fragments_dataset": "f",
        "seg_dataset_prefix": "/x/o.zarr/s",
        **extra,
    }
    return os.path.basename(get_seg_config(step, "mws")["db"]["db_file"])


def test_seeded_and_unseeded_runs_default_to_other_databases():
    assert default_db() != default_db(seeds_dataset="/x/seeds")


def test_the_default_seeded_names_fit_in_a_file_name():
    assert len(default_db(seeds_dataset="/x/seeds").encode()) < 255
    assert (
        len(
            mws_segmentation_name(
                {**DEFAULTS["mws"], "seeds_dataset": "/x/seeds"}
            ).encode()
        )
        < 255
    )


def test_seeds_with_another_method_are_refused():
    with pytest.raises(ValueError, match="seeds_dataset"):
        get_seg_config(
            {"affs_dataset": "a", "seg_dataset_prefix": "/x/s", "seeds_dataset": "s"},
            "ws",
        )


def test_other_seeds_are_named_in_the_overwrite_warning(tmp_path):
    store = str(tmp_path / "v.zarr" / "frags")
    zarr.open(store, mode="w", shape=(2, 2), dtype="u1")
    dump_params(store, {"affs_dataset": "/a/affs", "seeds_dataset": "/s/one"})

    warning = inputs_differ(
        store, {"affs_dataset": "/a/affs", "seeds_dataset": "/s/two"}
    )

    assert "/s/one" in warning and "/s/two" in warning


def mws_step(tmp_path, affs, **extra):
    """A one-block mws step on the conftest affs: 3 channels, 8 x 64 x 64 voxels of 50 x 8 x 8 nm."""
    return {
        "affs_dataset": affs,
        "fragments_dataset": f"{tmp_path}/v.zarr/fragments_mws",
        "seg_dataset_prefix": f"{tmp_path}/v.zarr/segmentations_mws",
        "params": {"aff_neighborhood": NB[:3], "bias": [-0.5] * 3, "strides": None},
        **extra,
    }


def test_float_seeds_are_refused_by_name(tmp_path, affs):
    seeds = f"{tmp_path}/v.zarr/seeds"
    prepare_ds(seeds, shape=(8, 64, 64), voxel_size=(50, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="float32") as refused:
        mutex_watershed_segmentation(
            get_seg_config(mws_step(tmp_path, affs, seeds_dataset=seeds), "mws")
        )

    assert seeds in str(refused.value)


def test_the_roi_is_cut_to_the_affs_and_snapped_to_their_grid(tmp_path, affs, caplog):
    run_segmentation(
        mws_step(tmp_path, affs, roi_offset=[10, 4, 4], roi_shape=[200, 600, 100]),
        "mws",
    )

    [seg] = glob.glob(f"{tmp_path}/v.zarr/segmentations_mws/*")
    params = zarr.open(seg, mode="r").attrs["bs_params"]
    assert (params["roi_offset"], params["roi_shape"]) == ([0, 0, 0], [250, 512, 104])
    assert "[0:250, 0:512, 0:104]" in caplog.text


def test_a_roi_outside_the_affs_is_refused_by_name(tmp_path, affs):
    with pytest.raises(ValueError, match="outside") as refused:
        run_segmentation(
            mws_step(
                tmp_path, affs, roi_offset=[0, 0, 4000], roi_shape=[400, 512, 512]
            ),
            "mws",
        )

    assert affs in str(refused.value)
