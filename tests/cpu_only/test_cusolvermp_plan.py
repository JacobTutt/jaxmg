import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxmg import plan_potrs_cusolvermp


def test_plan_potrs_cusolvermp_combines_runtime_and_potrs_preview():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    plan = plan_potrs_cusolvermp(a, b, 4, mesh=mesh, in_specs=(P("x", None),))

    assert plan.runtime.backend_family == "mp"
    assert plan.runtime.requires_mpi is True
    assert plan.runtime.requires_nccl is True
    assert plan.runtime.max_nrhs == 1
    assert plan.runtime.runtime.global_device_count == mesh.devices.size
    assert plan.runtime.runtime.process_grid == (1, 1)
    assert plan.potrs.current.axis_name == "x"
    expected_padding = 2 if mesh.devices.size == 1 else 1
    assert plan.potrs.current.padding == expected_padding
    assert plan.potrs.cyclic_2d.block_shape == (4, 4)
    assert plan.potrs.cyclic_2d.process_grid == plan.runtime.runtime.process_grid
