"""The catalog accepts the 9 chains that make one round and refuses the others with the fix."""

import pytest
from click import ClickException

from bootstrapper.catalog import check_round, parse_chain, valid_chains


def test_nine_chains_make_one_round():
    chains = [" ".join(m.short for m in chain) for chain in valid_chains()]

    assert chains == ["2A 3Af2A", "2L 3Af2L", "2M 3Af2A", "2M 3Af2L", "2M 3Af2M", "3A", "3L 3Af3L", "3M", "3M 3Af3L"]
    for text in chains:
        check_round(parse_chain(text))


@pytest.mark.parametrize(
    "text, refusal",
    [
        (
            "2d_lsd 3Af2M",
            "3d_affs_from_2d_mtlsd reads 2d_lsds and 2d_affs; 2d_lsd gives 2d_lsds. "
            "Models that read 2d_lsd: 3d_affs_from_2d_lsd (3Af2L)",
        ),
        (
            "2m 3af2l 3m 3af3l",
            "3d_mtlsd trains on this round's segmentation, so it starts the next round. "
            "This round: 2d_mtlsd → 3d_affs_from_2d_lsd; next round: 3d_mtlsd → 3d_affs_from_3d_lsd",
        ),
        (
            "2d_mtlsd",
            "2d_mtlsd gives 2d_lsds and 2d_affs; a round ends with 3D affinities. Add one after it: "
            "3d_affs_from_2d_affs (3Af2A), 3d_affs_from_2d_lsd (3Af2L), 3d_affs_from_2d_mtlsd (3Af2M)",
        ),
    ],
)
def test_a_chain_outside_one_round_is_refused_with_the_fix(text, refusal):
    with pytest.raises(ClickException) as e:
        check_round(parse_chain(text))

    assert e.value.message == refusal
