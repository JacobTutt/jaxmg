import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxmg import build_potrs_cusolvermp_call_spec


def test_build_potrs_cusolvermp_call_spec_tracks_matrix_and_rhs_layouts():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    spec = build_potrs_cusolvermp_call_spec(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )

    assert spec.ffi_target_name == "potrs_cusolvermp"
    assert spec.implementation_ready is False
    assert spec.executable is False
    assert spec.backend_family == "mp"
    assert spec.distribution == "2d-block-cyclic"
    assert spec.local_layout == "column-major"
    assert spec.contract_supported is True
    assert spec.current_rhs_shape == (6,)
    assert spec.normalized_rhs_shape == (6, 1)
    assert spec.matrix.block_shape == (4, 4)
    assert spec.matrix.global_shape == (6, 6)
    assert spec.rhs is not None
    assert spec.rhs.block_shape == (4, 1)
    assert spec.rhs.global_shape == (6, 1)


def test_build_potrs_cusolvermp_call_spec_carries_contract_failures():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6, 3))

    spec = build_potrs_cusolvermp_call_spec(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )

    assert spec.contract_supported is False
    assert spec.executable is False
    assert spec.rhs is not None
    assert "NRHS=1" in spec.contract_failure_reason
