from dataclasses import dataclass
from typing import Tuple

from jax.sharding import Mesh

from ._cusolvermp_runtime import CuSolverMpRuntimePlan, plan_cusolvermp_runtime
from ._cusolvermp_targets import cusolvermp_target_specs


@dataclass(frozen=True)
class CuSolverMpTargetPlan:
    operation: str
    ffi_target_name: str
    library_name: str
    symbol_name: str
    implemented: bool


@dataclass(frozen=True)
class CuSolverMpBackendPlan:
    backend_family: str
    distribution: str
    local_layout: str
    runtime: CuSolverMpRuntimePlan
    targets: Tuple[CuSolverMpTargetPlan, ...]
    unsupported_operations: Tuple[str, ...]


def plan_cusolvermp_backend(
    mesh: Mesh, process_grid: Tuple[int, int] | None = None
) -> CuSolverMpBackendPlan:
    """Capture the future backend-level cuSOLVERMp integration contract."""
    runtime = plan_cusolvermp_runtime(mesh, process_grid=process_grid)
    targets = tuple(
        CuSolverMpTargetPlan(
            operation=spec.operation,
            ffi_target_name=spec.ffi_target_name,
            library_name=spec.library_name,
            symbol_name=spec.symbol_name,
            implemented=False,
        )
        for spec in cusolvermp_target_specs()
    )
    return CuSolverMpBackendPlan(
        backend_family="mp",
        distribution="2d-block-cyclic",
        local_layout="column-major",
        runtime=runtime,
        targets=targets,
        unsupported_operations=("potri",),
    )
