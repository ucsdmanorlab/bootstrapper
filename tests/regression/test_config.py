"""Config files are kind tables; the loader keeps each step's tables and refuses the rest by name."""

import click
import pytest

from bootstrapper.config import load


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
            'affs_dataset = "/a"\nblockwise = true\n[mws_params]\nbias = [-0.5]\n',
            ["top level", "[segment]", "[segment.params]"],
            id="flat pre-1.0 file",
        ),
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
