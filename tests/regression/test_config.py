"""Config files are kind tables; the loader keeps each step's tables and refuses the rest by
name. Every table refuses an unknown key with the nearest valid name or the list."""

import glob

import click
import numpy as np
import pytest
from click.testing import CliRunner
from funlib.persistence import prepare_ds

from bootstrapper.cli import cli
from bootstrapper.config import load

SEGMENT = """
[segment]
affs_dataset = "/a/v.zarr/3d_affs"
fragments_dataset = "/a/o.zarr/frags"
seg_dataset_prefix = "/a/o.zarr/segs"
method = "mws"
"""


@pytest.fixture
def write(tmp_path):
    def write(text):
        path = tmp_path / "x.toml"
        path.write_text(text)
        return str(path)

    return write


def test_a_round_file_gives_its_steps_in_file_order(write):
    steps = load(
        write("""
[volumes.apical]
raw = "/v/apical.zarr/raw/s1"
labels = "/v/apical.zarr/labels/s1"
[volumes.spine]
raw = "/v/spine.zarr/raw/s0"
[train.setup_01]
model = "2d_mtlsd"
max_iterations = 30001
my_custom_key = 3
[predict.setup_01]
checkpoint = "/r/setups/setup_01/model_checkpoint_30000.ckpt"
[predict.3d_affs]
setup_dir = "/m/3d_affs_from_2d_mtlsd"
checkpoint = "/m/3d_affs_from_2d_mtlsd/model_checkpoint_25000.ckpt"
[segment]
method = "mws"
[segment.params]
global_bias = [1.0, -0.5]
[segment.db]
db_file = "/a/rag.db"
[segment.spine]
mask_dataset = "/v/spine.zarr/hand_mask/s0"
[refine]
[refine.size-filter]
min_size = 64
[refine.fill-holes]
xy = true
[evaluate]
thresholds = [0.1, 1.0]
""")
    )
    train, segment, refine = steps[0], steps[3], steps[4]

    assert [(s.kind, s.name) for s in steps] == [
        ("train", "setup_01"),
        ("predict", "setup_01"),
        ("predict", "3d_affs"),
        ("segment", "segment"),
        ("refine", "refine"),
        ("evaluate", "evaluate"),
    ]
    assert segment.keys["params"] == {"global_bias": [1.0, -0.5]}
    assert segment.keys["db"] == {"db_file": "/a/rag.db"}
    assert segment.volumes == {"spine": {"mask_dataset": "/v/spine.zarr/hand_mask/s0"}}
    assert "spine" not in segment.keys
    assert list(refine.keys["ops"]) == ["size-filter", "fill-holes"]
    assert refine.keys["ops"]["fill-holes"] == {"xy": True}
    assert train.keys["my_custom_key"] == 3


def test_named_steps_keep_their_params_and_volume_tables(write):
    steps = load(
        write("""
[volumes.b]
raw = "/v/b.zarr/raw"
[segment.mito]
method = "ws"
[segment.mito.params]
thresholds = [0.2, 0.3]
[segment.mito.b]
blockwise = false
[segment.synapse]
method = "cc"
""")
    )

    assert [(s.name, s.keys["method"], s.volumes) for s in steps] == [
        ("mito", "ws", {"b": {"blockwise": False}}),
        ("synapse", "cc", {}),
    ]
    assert steps[0].keys["params"] == {"thresholds": [0.2, 0.3]}


@pytest.mark.parametrize(
    "text,needles",
    [
        pytest.param(
            'setup_dir = "/s"\nmax_iterations = 3\n[[samples]]\nraw = "/r"\n',
            ["[train.<name>]"],
            id="flat train file",
        ),
        pytest.param(
            '[segment]\nmethod = "mws"\n[segment.mws_params]\nbias = [-0.5]\n',
            ["mws_params", "[segment.params]"],
            id="old params table",
        ),
        pytest.param(
            '[segment]\nmethod = "mws"\nblokwise = true\n',
            ["blokwise", "blockwise"],
            id="unknown segment key",
        ),
        pytest.param(
            '[segmnt]\nmethod = "mws"\n', ["segmnt", "segment"], id="unknown table"
        ),
        pytest.param(
            '[volumes.a]\nraw = "/r"\n[segment.a]\nmethod = "mws"\n[segment.b]\nmethod = "cc"\n',
            ["[segment.a]", "volume 'a'"],
            id="step named like a volume",
        ),
        pytest.param(
            "[refine]\n[refine.blur]\nsigma = 1\n",
            ["blur", "fill-holes"],
            id="unknown refine op",
        ),
    ],
)
def test_a_bad_file_is_refused_by_name(write, text, needles):
    with pytest.raises(click.ClickException) as error:
        load(write(text))

    for needle in needles:
        assert needle in str(error.value)


@pytest.fixture
def labels(tmp_path):
    """Two objects of 4x32x16 voxels in t.zarr/labels."""
    path = str(tmp_path / "t.zarr" / "labels")
    ds = prepare_ds(
        path,
        shape=(4, 32, 32),
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint64,
    )
    data = np.ones((4, 32, 32), np.uint64)
    data[:, :, 16:] = 2
    ds[ds.roi] = data
    return path


@pytest.mark.parametrize(
    "text,message",
    [
        pytest.param(
            '[volumes.spine]\nraw = "/v/spine.zarr/raw/s0"\n[segment]\nmethod = "mws"\n'
            "[segment.spine]\nblokwise = true\n",
            "[segment.spine] has no key 'blokwise'; did you mean 'blockwise'?",
            id="typo in a volume table",
        ),
        pytest.param(
            SEGMENT + "num_workers = 20\nblockwise = true\n"
            '[segment.db]\ndb_name = "x"\ndb_host = "localhost"\ndb_user = "vijay"\ndb_password = ""\n'
            'db_port = 5433\nnodes_table = "nodes"\nedges_table = "edges"\n'
            "[segment.params]\nglobal_bias = [1.0, -0.5]\n",
            "[segment.db] has no key 'db_port'; volara has no port setting, so set PGPORT in the environment",
            id="db port",
        ),
        pytest.param(
            'affs_dataset = "/a/v.zarr/3d_affs"\nfragments_dataset = "/a/o.zarr/frags"\n'
            'seg_dataset_prefix = "/a/o.zarr/segs"\nblockwise = true\n'
            "[mws_params]\nbias = [-0.5, -0.5, -0.5]\n",
            "keys at the top level; wrap them in [segment] (or [evaluate], [train.<name>], "
            "[predict.<name>]), and rename [mws_params] to [segment.params]",
            id="flat pre-1.0 file",
        ),
        pytest.param(
            "[mws_params]\nbias = [-0.5]\n",
            "rename [mws_params] to [segment.params]",
            id="old params table at the top",
        ),
    ],
)
def test_the_loader_refuses_an_unknown_key_with_the_fix(write, text, message):
    path = write(text)

    with pytest.raises(click.ClickException) as error:
        load(path)

    assert str(error.value) == f"{path}: {message}"


@pytest.mark.parametrize(
    "params,args,message",
    [
        pytest.param(
            "seed_eps = 0.5\nremove_debris = 32\n",
            [],
            "[segment.params] for mws has no key 'seed_eps'; keys are aff_neighborhood, bias, "
            "filter_fragments, global_bias, min_seed_distance, noise_eps, randomized_strides, "
            "remove_debris, sigma, strides",
            id="a ws key under mws",
        ),
        pytest.param(
            "remove_debri = 32\n",
            [],
            "[segment.params] for mws has no key 'remove_debri'; did you mean 'remove_debris'?",
            id="typo in params",
        ),
        pytest.param(
            "remove_debris = 32\n",
            ["-p", "remove_debri=32"],
            "-p for mws has no key 'remove_debri'; did you mean 'remove_debris'?",
            id="typo in -p",
        ),
    ],
)
def test_segment_refuses_an_unknown_param_before_it_runs(write, params, args, message):
    path = write(SEGMENT + "[segment.params]\n" + params)

    result = CliRunner().invoke(cli, ["segment", path, *args])

    assert result.exit_code == 1
    assert f"Error: {message}" in result.stderr.splitlines()
    assert "SQLite" not in result.output


@pytest.mark.parametrize(
    "ops,message",
    [
        pytest.param(
            "[refine.size-filter]\nmin_size = 8\ndry_run = true\nnum_workers = 1\n"
            "[refine.fill-holes]\nxy = true\nnum_workers = 1\n",
            "[refine.size-filter] has no key 'dry_run'; a chain has no dry run, "
            "so run bs refine size-filter --dry-run",
            id="dry_run on a filter",
        ),
        pytest.param(
            "[refine.size-filter]\nmin_sz = 8\nnum_workers = 1\n",
            "[refine.size-filter] has no key 'min_sz'; did you mean 'min_size'?",
            id="typo in an op",
        ),
    ],
)
def test_a_refine_chain_refuses_an_unknown_op_key_before_any_op_runs(
    write, labels, ops, message
):
    path = write(f'[refine]\nin_array = "{labels}"\n' + ops)

    result = CliRunner().invoke(cli, ["run", path])

    assert result.exit_code == 1
    assert f"Error: {message}" in result.stderr.splitlines()
    assert "objects" not in result.output
    assert glob.glob(f"{labels}_*") == []


def test_a_valid_kind_table_file_still_loads(write):
    segment, refine = load(
        write("""
[volumes.spine]
raw = "/v/spine.zarr/raw/s0"
[segment]
method = "mws"
blockwise = true
[segment.params]
remove_debris = 32
[segment.db]
db_name = "x"
db_host = "localhost"
db_user = "vijay"
db_password = ""
[segment.spine]
num_workers = 4
[segment.spine.db]
db_file = "/o/rag.db"
[refine]
in_array = "/o.zarr/segs"
[refine.size-filter]
min_size = 8
out_array = "/o.zarr/big"
[refine.fill-holes]
xy = true
""")
    )

    assert segment.keys["db"]["db_host"] == "localhost"
    assert segment.volumes == {
        "spine": {"num_workers": 4, "db": {"db_file": "/o/rag.db"}}
    }
    assert refine.keys["ops"] == {
        "size-filter": {"min_size": 8, "out_array": "/o.zarr/big"},
        "fill-holes": {"xy": True},
    }
