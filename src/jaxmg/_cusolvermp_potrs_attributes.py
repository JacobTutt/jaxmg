import json
from dataclasses import dataclass
from typing import Any

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._cusolvermp_potrs_context import (
    CuSolverMpPotrsContextConfig,
    build_potrs_cusolvermp_context_config,
)


@dataclass(frozen=True)
class CuSolverMpPotrsAttributes:
    operation: str
    version: int
    payload: dict[str, Any]
    payload_json: str
    payload_bytes: bytes


def _build_potrs_cusolvermp_payload(
    config: CuSolverMpPotrsContextConfig,
) -> dict[str, Any]:
    payload = config.asdict()
    payload["operation"] = "potrs"
    payload["version"] = 1
    return payload


def encode_potrs_cusolvermp_attributes(
    config: CuSolverMpPotrsContextConfig,
) -> CuSolverMpPotrsAttributes:
    payload = _build_potrs_cusolvermp_payload(config)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return CuSolverMpPotrsAttributes(
        operation="potrs",
        version=1,
        payload=payload,
        payload_json=payload_json,
        payload_bytes=payload_json.encode("utf-8"),
    )


def build_potrs_cusolvermp_attributes(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: tuple[P] | list[P] | P,
    pad: bool = True,
    process_grid: tuple[int, int] | None = None,
) -> CuSolverMpPotrsAttributes:
    """Serialize the future cuSOLVERMp potrs context into a stable attribute payload."""
    config = build_potrs_cusolvermp_context_config(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    return encode_potrs_cusolvermp_attributes(config)
