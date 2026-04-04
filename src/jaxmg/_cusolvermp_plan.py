from dataclasses import dataclass
from typing import List, Tuple

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._distributed_runtime import DistributedRuntimePlan, plan_distributed_runtime
from ._potrs import _PotrsMigrationPreview, _plan_potrs_migration_preview


@dataclass(frozen=True)
class PotrsCuSolverMpPlan:
    runtime: DistributedRuntimePlan
    potrs: _PotrsMigrationPreview


def plan_potrs_cusolvermp(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P] | P,
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
) -> PotrsCuSolverMpPlan:
    """Build the solver-level planning object for a future cuSOLVERMp potrs path."""
    runtime = plan_distributed_runtime(mesh, process_grid=process_grid)
    potrs = _plan_potrs_migration_preview(
        a,
        b,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=runtime.process_grid,
    )
    return PotrsCuSolverMpPlan(runtime=runtime, potrs=potrs)
