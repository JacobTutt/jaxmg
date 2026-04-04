from dataclasses import dataclass
from typing import Tuple

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._block_cyclic_2d import local_block_cyclic_shape, padded_block_shape
from ._cusolvermp_plan import PotrsCuSolverMpPlan, plan_potrs_cusolvermp


@dataclass(frozen=True)
class CuSolverMpArrayCallLayout:
    global_shape: Tuple[int, int]
    block_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    padding: Tuple[int, int]
    local_shapes: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class CuSolverMpPotrsCallSpec:
    ffi_target_name: str
    library_name: str
    symbol_name: str
    implementation_ready: bool
    executable: bool
    backend_family: str
    distribution: str
    local_layout: str
    runtime_launch_ready: bool
    runtime_issues: Tuple[str, ...]
    contract_supported: bool
    contract_failure_reason: str | None
    requires_rhs_redistribution: bool
    current_rhs_shape: Tuple[int, ...]
    normalized_rhs_shape: Tuple[int, int] | None
    matrix: CuSolverMpArrayCallLayout
    rhs: CuSolverMpArrayCallLayout | None


def _build_rhs_call_layout(plan: PotrsCuSolverMpPlan) -> CuSolverMpArrayCallLayout | None:
    normalized_rhs_shape = plan.contract.normalized_rhs_shape
    if normalized_rhs_shape is None:
        return None

    mb = plan.potrs.cyclic_2d.block_shape[0]
    rhs_block_shape = (mb, 1)
    rhs_padded_shape, rhs_padding = padded_block_shape(
        normalized_rhs_shape, rhs_block_shape
    )
    local_shapes = tuple(
        local_block_cyclic_shape(
            rhs_padded_shape,
            rhs_block_shape,
            plan.potrs.cyclic_2d.process_grid,
            process_coords,
        )
        for process_coords in plan.potrs.cyclic_2d.rank_order
    )
    return CuSolverMpArrayCallLayout(
        global_shape=normalized_rhs_shape,
        block_shape=rhs_block_shape,
        padded_shape=rhs_padded_shape,
        padding=rhs_padding,
        local_shapes=local_shapes,
    )


def build_potrs_cusolvermp_call_spec(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | list[P] | P,
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
) -> CuSolverMpPotrsCallSpec:
    """Build a non-executing call specification for the future cuSOLVERMp potrs path."""
    plan = plan_potrs_cusolvermp(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )
    target = next(
        target for target in plan.backend.targets if target.operation == "potrs"
    )

    matrix = CuSolverMpArrayCallLayout(
        global_shape=plan.potrs.cyclic_2d.global_shape,
        block_shape=plan.potrs.cyclic_2d.block_shape,
        padded_shape=plan.potrs.cyclic_2d.padded_shape,
        padding=plan.potrs.cyclic_2d.padding,
        local_shapes=plan.potrs.cyclic_2d.local_shapes,
    )
    rhs = _build_rhs_call_layout(plan)
    runtime_ready = plan.backend.runtime.readiness.launch_ready
    contract_supported = plan.contract.supported

    return CuSolverMpPotrsCallSpec(
        ffi_target_name=target.ffi_target_name,
        library_name=target.library_name,
        symbol_name=target.symbol_name,
        implementation_ready=target.implemented,
        executable=target.implemented and runtime_ready and contract_supported,
        backend_family=plan.backend.backend_family,
        distribution=plan.backend.distribution,
        local_layout=plan.backend.local_layout,
        runtime_launch_ready=runtime_ready,
        runtime_issues=plan.backend.runtime.readiness.issues,
        contract_supported=contract_supported,
        contract_failure_reason=plan.contract.failure_reason,
        requires_rhs_redistribution=True,
        current_rhs_shape=plan.contract.rhs_shape,
        normalized_rhs_shape=plan.contract.normalized_rhs_shape,
        matrix=matrix,
        rhs=rhs,
    )
