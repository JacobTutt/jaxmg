import pytest
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxmg import build_potrs_cusolvermp_stub_request, potrs_cusolvermp


def test_build_potrs_cusolvermp_stub_request_matches_future_target():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    request = build_potrs_cusolvermp_stub_request(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )

    assert request.context.ffi_target_name == "potrs_cusolvermp"
    assert request.attributes.payload["ffi_target_name"] == "potrs_cusolvermp"
    assert request.attributes.payload["operation"] == "potrs"


def test_potrs_cusolvermp_raises_explicit_stub_error():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    with pytest.raises(NotImplementedError, match="requires a GPU runtime"):
        potrs_cusolvermp(a, b, 4, mesh=mesh, in_specs=(P("x", None),))
