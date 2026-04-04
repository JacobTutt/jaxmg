from dataclasses import dataclass
from typing import Tuple

from jax.sharding import Mesh

from ._distributed_runtime import DistributedRuntimePlan, plan_distributed_runtime


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


def plan_cusolvermp_runtime(
    mesh: Mesh, process_grid: Tuple[int, int] | None = None
) -> CuSolverMpRuntimePlan:
    """Capture the fixed runtime assumptions for the future cuSOLVERMp backend."""
    runtime = plan_distributed_runtime(mesh, process_grid=process_grid)
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
    )
