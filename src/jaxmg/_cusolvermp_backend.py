from dataclasses import dataclass
from typing import Tuple

from jax.sharding import Mesh

from ._cusolvermp_runtime import CuSolverMpRuntimePlan, plan_cusolvermp_runtime


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
    targets = (
        CuSolverMpTargetPlan(
            operation="potrs",
            ffi_target_name="potrs_cusolvermp",
            library_name="libpotrs_cusolvermp.so",
            symbol_name="PotrsCuSolverMpFFI",
            implemented=False,
        ),
        CuSolverMpTargetPlan(
            operation="syevd",
            ffi_target_name="syevd_cusolvermp",
            library_name="libsyevd_cusolvermp.so",
            symbol_name="SyevdCuSolverMpFFI",
            implemented=False,
        ),
    )
    return CuSolverMpBackendPlan(
        backend_family="mp",
        distribution="2d-block-cyclic",
        local_layout="column-major",
        runtime=runtime,
        targets=targets,
        unsupported_operations=("potri",),
    )
