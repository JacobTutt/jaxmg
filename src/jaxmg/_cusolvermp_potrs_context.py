from dataclasses import asdict, dataclass
from typing import Any, Tuple

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._cusolvermp_plan import plan_potrs_cusolvermp
from ._cusolvermp_potrs_call import (
    CuSolverMpPotrsCallSpec,
    build_potrs_cusolvermp_call_spec,
)


@dataclass(frozen=True)
class CuSolverMpPotrsContextConfig:
    ffi_target_name: str
    library_name: str
    symbol_name: str
    process_rank: int
    process_count: int
    local_device_count: int
    local_device_index: int
    global_device_count: int
    process_grid: Tuple[int, int]
    matrix_block_shape: Tuple[int, int]
    rhs_block_shape: Tuple[int, int] | None
    matrix_padded_shape: Tuple[int, int]
    rhs_padded_shape: Tuple[int, int] | None
    requires_mpi: bool
    requires_nccl: bool
    launch_ready: bool
    launch_issues: Tuple[str, ...]
    contract_supported: bool
    contract_failure_reason: str | None
    requires_rhs_redistribution: bool
    current_rhs_shape: Tuple[int, ...]
    normalized_rhs_shape: Tuple[int, int] | None
    implementation_ready: bool
    executable: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def build_potrs_cusolvermp_context_config(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | list[P] | P,
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
) -> CuSolverMpPotrsContextConfig:
    """Build the future native cuSOLVERMp context payload for potrs."""
    plan = plan_potrs_cusolvermp(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    call_spec = build_potrs_cusolvermp_call_spec(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )

    return CuSolverMpPotrsContextConfig(
        ffi_target_name=call_spec.ffi_target_name,
        library_name=call_spec.library_name,
        symbol_name=call_spec.symbol_name,
        process_rank=plan.backend.runtime.runtime.process_index,
        process_count=plan.backend.runtime.runtime.process_count,
        local_device_count=plan.backend.runtime.runtime.local_device_count,
        local_device_index=0,
        global_device_count=plan.backend.runtime.runtime.global_device_count,
        process_grid=plan.backend.runtime.runtime.process_grid,
        matrix_block_shape=call_spec.matrix.block_shape,
        rhs_block_shape=None if call_spec.rhs is None else call_spec.rhs.block_shape,
        matrix_padded_shape=call_spec.matrix.padded_shape,
        rhs_padded_shape=None if call_spec.rhs is None else call_spec.rhs.padded_shape,
        requires_mpi=plan.backend.runtime.requires_mpi,
        requires_nccl=plan.backend.runtime.requires_nccl,
        launch_ready=call_spec.runtime_launch_ready,
        launch_issues=call_spec.runtime_issues,
        contract_supported=call_spec.contract_supported,
        contract_failure_reason=call_spec.contract_failure_reason,
        requires_rhs_redistribution=call_spec.requires_rhs_redistribution,
        current_rhs_shape=call_spec.current_rhs_shape,
        normalized_rhs_shape=call_spec.normalized_rhs_shape,
        implementation_ready=call_spec.implementation_ready,
        executable=call_spec.executable,
    )
