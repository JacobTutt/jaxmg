import os

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from jaxmg._potrs import _plan_potrs_migration_preview


@pytest.fixture(autouse=True)
def _set_device_count(monkeypatch):
    monkeypatch.setenv("JAXMG_NUMBER_OF_DEVICES", "2")


def test_potrs_migration_preview_keeps_current_and_future_views():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))
    preview = _plan_potrs_migration_preview(a, b, 4, mesh=mesh, in_specs=(P("x", None),))

    assert preview.current.axis_name == "x"
    assert preview.current.shard_size == 3
    assert preview.current.padding == 1

    assert preview.cyclic_2d.global_shape == (6, 6)
    assert preview.cyclic_2d.block_shape == (4, 4)
    assert preview.cyclic_2d.padded_shape == (8, 8)
    assert preview.cyclic_2d.padding == (2, 2)
    assert preview.cyclic_2d.process_grid == (1, 1)


def test_potrs_migration_preview_accepts_explicit_process_grid():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(4)
    b = jnp.ones((4, 1))
    preview = _plan_potrs_migration_preview(
        a, b, 2, mesh=mesh, in_specs=(P("x", None),), process_grid=(1, 1)
    )
    assert preview.cyclic_2d.process_grid == (1, 1)
