import jax
import jax.numpy as jnp
import pytest

from jaxmg import plan_distributed_runtime


def test_plan_distributed_runtime_uses_mesh_device_count():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    plan = plan_distributed_runtime(mesh)
    assert plan.process_count == jax.process_count()
    assert plan.process_index == jax.process_index()
    assert plan.local_device_count == jax.local_device_count()
    assert plan.local_device_index == (
        getattr(jax.local_devices()[0], "local_hardware_id", None) or 0
    )
    assert plan.global_device_count == jax.device_count()
    assert plan.process_grid == (1, 1)


def test_plan_distributed_runtime_accepts_explicit_process_grid():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    plan = plan_distributed_runtime(mesh, process_grid=(1, 1))
    assert plan.process_grid == (1, 1)


def test_plan_distributed_runtime_rejects_bad_grid():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(ValueError, match="nprow \\* npcol == num_devices"):
        plan_distributed_runtime(mesh, process_grid=(1, 2))
