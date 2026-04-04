import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxmg import build_potrs_cusolvermp_context_config


def test_build_potrs_cusolvermp_context_config_tracks_native_payload_fields():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    config = build_potrs_cusolvermp_context_config(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )

    assert config.ffi_target_name == "potrs_cusolvermp"
    assert config.process_rank == 0
    assert config.process_count == jax.process_count()
    assert config.local_device_count == jax.local_device_count()
    assert config.local_device_index == 0
    assert config.global_device_count == mesh.devices.size
    assert config.process_grid == (1, 1)
    assert config.matrix_block_shape == (4, 4)
    assert config.rhs_block_shape == (4, 1)
    assert config.matrix_padded_shape == (8, 8)
    assert config.rhs_padded_shape == (8, 1)
    assert config.requires_mpi is True
    assert config.requires_nccl is True
    assert config.contract_supported is True
    assert config.implementation_ready is False
    assert config.executable is False
    assert config.asdict()["ffi_target_name"] == "potrs_cusolvermp"


def test_build_potrs_cusolvermp_context_config_preserves_rhs_contract_failures():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6, 3))

    config = build_potrs_cusolvermp_context_config(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )

    assert config.contract_supported is False
    assert "NRHS=1" in config.contract_failure_reason
    assert config.executable is False
