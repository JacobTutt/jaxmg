from dataclasses import dataclass
from typing import Tuple

from jax.sharding import Mesh

from ._distributed_runtime import DistributedRuntimePlan, plan_distributed_runtime


@dataclass(frozen=True)
class CuSolverMpRuntimeReadiness:
    launch_ready: bool
    issues: Tuple[str, ...]
    expected_process_count: int
    expected_local_device_count: int


@dataclass(frozen=True)
class CuSolverMpRuntimePlan:
    backend_family: str
    launch_model: str
    requires_mpi: bool
    requires_nccl: bool
    supports_multi_node: bool
    supports_potri: bool
    supports_multi_rhs: bool
    max_nrhs: int
    runtime: DistributedRuntimePlan
    readiness: CuSolverMpRuntimeReadiness


def _assess_cusolvermp_runtime_readiness(
    runtime: DistributedRuntimePlan,
) -> CuSolverMpRuntimeReadiness:
    issues = []

    if runtime.process_count != runtime.global_device_count:
        issues.append(
            "cuSOLVERMp expects one JAX process per GPU, so "
            "jax.process_count() must equal the global device count."
        )

    if runtime.local_device_count != 1:
        issues.append(
            "cuSOLVERMp expects each process to control exactly one local GPU."
        )

    return CuSolverMpRuntimeReadiness(
        launch_ready=not issues,
        issues=tuple(issues),
        expected_process_count=runtime.global_device_count,
        expected_local_device_count=1,
    )


def plan_cusolvermp_runtime(
    mesh: Mesh, process_grid: Tuple[int, int] | None = None
) -> CuSolverMpRuntimePlan:
    """Capture the fixed runtime assumptions for the future cuSOLVERMp backend."""
    runtime = plan_distributed_runtime(mesh, process_grid=process_grid)
    readiness = _assess_cusolvermp_runtime_readiness(runtime)
    return CuSolverMpRuntimePlan(
        backend_family="mp",
        launch_model="one-process-per-gpu",
        requires_mpi=True,
        requires_nccl=True,
        supports_multi_node=True,
        supports_potri=False,
        supports_multi_rhs=False,
        max_nrhs=1,
        runtime=runtime,
        readiness=readiness,
    )
