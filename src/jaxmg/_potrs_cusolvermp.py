import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Tuple

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._setup import ensure_init_jaxmg_backend
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


def _build_native_stub_attrs(
    request: CuSolverMpPotrsStubRequest, T_A: int
) -> dict[str, int | str]:
    rhs_block_shape = request.context.rhs_block_shape or (-1, -1)
    rhs_padded_shape = request.context.rhs_padded_shape or (-1, -1)
    return {
        "T_A": T_A,
        "context_json": request.attributes.payload_json,
        "process_rank": request.context.process_rank,
        "process_count": request.context.process_count,
        "local_device_count": request.context.local_device_count,
        "local_device_index": request.context.local_device_index,
        "global_device_count": request.context.global_device_count,
        "process_grid_nprow": request.context.process_grid[0],
        "process_grid_npcol": request.context.process_grid[1],
        "matrix_block_rows": request.context.matrix_block_shape[0],
        "matrix_block_cols": request.context.matrix_block_shape[1],
        "matrix_padded_rows": request.context.matrix_padded_shape[0],
        "matrix_padded_cols": request.context.matrix_padded_shape[1],
        "rhs_block_rows": rhs_block_shape[0],
        "rhs_block_cols": rhs_block_shape[1],
        "rhs_padded_rows": rhs_padded_shape[0],
        "rhs_padded_cols": rhs_padded_shape[1],
    }


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
    return_status: bool = False,
):
    out_a, out_b, status = _potrs_cusolvermp_raw(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    del out_a

    if b.ndim == 1:
        out_b = jnp.squeeze(out_b, axis=1)

    if return_status:
        return out_b, status[0]
    return out_b


def _potrs_cusolvermp_raw(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P] | P,
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
):
    """Stub entry seam for the future cuSOLVERMp potrs path.

    This first attempts to build the full future request path. When the
    experimental `mp` stub backend is available on a GPU runtime, it calls the
    registered native stub target. Otherwise it fails in Python with an
    explicit message.
    """
    ensure_init_jaxmg_backend()
    request = build_potrs_cusolvermp_stub_request(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    if not request.context.contract_supported:
        raise ValueError(request.context.contract_failure_reason)
    if not request.context.launch_ready:
        raise NotImplementedError(
            "potrs_cusolvermp requires a one-process-per-GPU launch. "
            f"Current issues: {request.context.launch_issues}"
        )
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise NotImplementedError(
            "potrs_cusolvermp native stub requires a GPU runtime. "
            f"Prepared target {request.context.ffi_target_name!r}."
        )

    normalized_b = b if b.ndim == 2 else jnp.expand_dims(b, axis=1)
    out_type = (
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct(normalized_b.shape, normalized_b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    ffi_fn = jax.ffi.ffi_call(
        request.context.ffi_target_name,
        out_type,
        input_layouts=((0, 1), (1, 0)),
        output_layouts=((0, 1), (1, 0), (0,)),
        input_output_aliases={0: 0, 1: 1},
    )
    return ffi_fn(a, normalized_b, **_build_native_stub_attrs(request, T_A))
