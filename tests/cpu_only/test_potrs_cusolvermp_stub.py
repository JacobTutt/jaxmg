import pytest
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxmg import build_potrs_cusolvermp_stub_request, potrs_cusolvermp
from jaxmg._potrs_cusolvermp import _build_native_stub_attrs


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


def test_build_native_stub_attrs_exposes_scalar_context_fields():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    request = build_potrs_cusolvermp_stub_request(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )
    attrs = _build_native_stub_attrs(request, 4)

    assert attrs["T_A"] == 4
    assert attrs["process_rank"] == 0
    assert attrs["process_count"] == 1
    assert attrs["process_grid_nprow"] == request.context.process_grid[0]
    assert attrs["process_grid_npcol"] == request.context.process_grid[1]
    assert attrs["matrix_block_rows"] == request.context.matrix_block_shape[0]
    assert attrs["matrix_block_cols"] == request.context.matrix_block_shape[1]
    assert attrs["matrix_padded_rows"] == request.context.matrix_padded_shape[0]
    assert attrs["matrix_padded_cols"] == request.context.matrix_padded_shape[1]
    assert attrs["rhs_block_rows"] == request.context.rhs_block_shape[0]
    assert attrs["rhs_block_cols"] == request.context.rhs_block_shape[1]
    assert attrs["rhs_padded_rows"] == request.context.rhs_padded_shape[0]
    assert attrs["rhs_padded_cols"] == request.context.rhs_padded_shape[1]
    assert "\"operation\":\"potrs\"" in attrs["context_json"]


def test_potrs_cusolvermp_matches_potrs_vector_return_contract(monkeypatch):
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4,), dtype=jnp.float32)

    def fake_raw(*args, **kwargs):
        del args, kwargs
        return (
            a,
            jnp.array([[1.0], [0.5], [1.0 / 3.0], [0.25]], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.int32),
        )

    monkeypatch.setattr("jaxmg._potrs_cusolvermp._potrs_cusolvermp_raw", fake_raw)

    out = potrs_cusolvermp(a, b, 2, mesh=mesh, in_specs=(P("x", None),))
    assert out.shape == (4,)
    assert jnp.allclose(out, jnp.array([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=jnp.float32))


def test_potrs_cusolvermp_can_return_status(monkeypatch):
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    def fake_raw(*args, **kwargs):
        del args, kwargs
        return (
            a,
            jnp.array([[1.0], [0.5], [1.0 / 3.0], [0.25]], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.int32),
        )

    monkeypatch.setattr("jaxmg._potrs_cusolvermp._potrs_cusolvermp_raw", fake_raw)

    out, status = potrs_cusolvermp(
        a, b, 2, mesh=mesh, in_specs=(P("x", None),), return_status=True
    )
    assert out.shape == (4, 1)
    assert int(status) == 0
