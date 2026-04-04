import os

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from jaxmg._potrs import _plan_potrs_layout


@pytest.fixture(autouse=True)
def _set_device_count(monkeypatch):
    monkeypatch.setenv("JAXMG_NUMBER_OF_DEVICES", "2")


def test_plan_potrs_layout_expands_vector_rhs():
    a = jnp.eye(4)
    b = jnp.ones((4,))
    plan = _plan_potrs_layout(a, b, 2, (P("x", None),))
    assert plan.b.shape == (4, 1)
    assert plan.axis_name == "x"
    assert plan.shard_size == 2
    assert plan.padding == 0


def test_plan_potrs_layout_rejects_bad_specs():
    a = jnp.eye(4)
    b = jnp.ones((4, 1))
    with pytest.raises(ValueError, match="A must be sharded along the columns"):
        _plan_potrs_layout(a, b, 1, (P(None, "x"),))


def test_plan_potrs_layout_tracks_padding():
    a = jnp.eye(6)
    b = jnp.ones((6, 1))
    plan = _plan_potrs_layout(a, b, 4, (P("x", None),))
    assert plan.ndev == 2
    assert plan.N_rows == 6
    assert plan.N == 6
    assert plan.shard_size == 3
    assert plan.padding == 1
