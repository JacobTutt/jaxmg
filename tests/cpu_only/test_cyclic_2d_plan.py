import jax
import jax.numpy as jnp
import pytest

from jax.sharding import PartitionSpec as P
from jaxmg import (
    choose_process_grid,
    local_block_cyclic_shape,
    normalize_process_grid,
    numroc,
    plan_cyclic_2d_layout,
)


def test_choose_process_grid_prefers_squareish_factorization():
    assert choose_process_grid(1) == (1, 1)
    assert choose_process_grid(2) == (1, 2)
    assert choose_process_grid(4) == (2, 2)
    assert choose_process_grid(6) == (2, 3)
    assert choose_process_grid(8) == (2, 4)


def test_normalize_process_grid_rejects_mismatched_product():
    with pytest.raises(ValueError, match="nprow \\* npcol == num_devices"):
        normalize_process_grid(4, (1, 3))


def test_numroc_conserves_global_extent():
    counts = [numroc(10, 2, proc, 2) for proc in range(2)]
    assert counts == [6, 4]
    assert sum(counts) == 10


def test_local_block_cyclic_shape_matches_even_2x2_case():
    assert local_block_cyclic_shape((8, 8), (2, 2), (2, 2), (0, 0)) == (4, 4)
    assert local_block_cyclic_shape((8, 8), (2, 2), (2, 2), (1, 1)) == (4, 4)


def test_plan_cyclic_2d_layout_uses_row_sharded_input_contract():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(4)
    plan = plan_cyclic_2d_layout(a, 2, mesh=mesh, in_specs=(P("x", None),))
    assert plan.global_shape == (4, 4)
    assert plan.block_shape == (2, 2)
    assert plan.process_grid == (1, 1)
    assert plan.local_shapes == ((4, 4),)


def test_plan_cyclic_2d_layout_rejects_non_row_sharded_specs():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(4)
    with pytest.raises(ValueError, match="A must be sharded along the rows"):
        plan_cyclic_2d_layout(a, 2, mesh=mesh, in_specs=(P(None, "x"),))
