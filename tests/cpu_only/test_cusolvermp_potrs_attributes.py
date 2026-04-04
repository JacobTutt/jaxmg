import json

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxmg import (
    build_potrs_cusolvermp_attributes,
    build_potrs_cusolvermp_context_config,
    encode_potrs_cusolvermp_attributes,
)


def test_encode_potrs_cusolvermp_attributes_is_stable_and_round_trips():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6,))

    config = build_potrs_cusolvermp_context_config(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )
    attrs = encode_potrs_cusolvermp_attributes(config)

    assert attrs.operation == "potrs"
    assert attrs.version == 1
    assert attrs.payload["ffi_target_name"] == "potrs_cusolvermp"
    assert attrs.payload_json == attrs.payload_bytes.decode("utf-8")
    assert json.loads(attrs.payload_json)["operation"] == "potrs"
    assert json.loads(attrs.payload_json)["version"] == 1


def test_build_potrs_cusolvermp_attributes_preserves_contract_failures():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    a = jnp.eye(6)
    b = jnp.ones((6, 3))

    attrs = build_potrs_cusolvermp_attributes(
        a, b, 4, mesh=mesh, in_specs=(P("x", None),)
    )

    decoded = json.loads(attrs.payload_json)
    assert decoded["contract_supported"] is False
    assert "NRHS=1" in decoded["contract_failure_reason"]
