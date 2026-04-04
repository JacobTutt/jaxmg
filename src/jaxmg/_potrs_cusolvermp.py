from dataclasses import dataclass
from typing import List, Tuple

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._cusolvermp_potrs_attributes import (
    CuSolverMpPotrsAttributes,
    build_potrs_cusolvermp_attributes,
)
from ._cusolvermp_potrs_context import (
    CuSolverMpPotrsContextConfig,
    build_potrs_cusolvermp_context_config,
)


@dataclass(frozen=True)
class CuSolverMpPotrsStubRequest:
    context: CuSolverMpPotrsContextConfig
    attributes: CuSolverMpPotrsAttributes


def build_potrs_cusolvermp_stub_request(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P] | P,
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
) -> CuSolverMpPotrsStubRequest:
    """Build the full future Python-side request for a cuSOLVERMp potrs call."""
    context = build_potrs_cusolvermp_context_config(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    attributes = build_potrs_cusolvermp_attributes(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    return CuSolverMpPotrsStubRequest(context=context, attributes=attributes)


def potrs_cusolvermp(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P] | P,
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
):
    """Stub entry seam for the future cuSOLVERMp potrs path.

    This intentionally does not execute a native FFI target yet. It exists so
    the Python-side path can build the eventual call request end-to-end and
    then fail in a single explicit place until the native backend is linked.
    """
    request = build_potrs_cusolvermp_stub_request(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    raise NotImplementedError(
        "potrs_cusolvermp is not linked yet. "
        f"Prepared target {request.context.ffi_target_name!r} with "
        f"launch_ready={request.context.launch_ready}, "
        f"contract_supported={request.context.contract_supported}."
    )
