from dataclasses import dataclass
from typing import Tuple

import jax
from jax.sharding import Mesh

from ._block_cyclic_2d import normalize_process_grid


@dataclass(frozen=True)
class DistributedRuntimePlan:
    process_count: int
    process_index: int
    local_device_count: int
    global_device_count: int
    process_grid: Tuple[int, int]


def plan_distributed_runtime(
    mesh: Mesh, process_grid: Tuple[int, int] | None = None
) -> DistributedRuntimePlan:
    """Capture the distributed runtime facts the future cuSOLVERMp path will need."""
    global_device_count = mesh.devices.size
    process_count = jax.process_count()
    process_index = jax.process_index()
    local_device_count = jax.local_device_count()

    if process_count <= 0:
        raise ValueError("jax.process_count() must be positive.")
    if local_device_count <= 0:
        raise ValueError("jax.local_device_count() must be positive.")
    if global_device_count <= 0:
        raise ValueError("mesh must contain at least one device.")

    process_grid = normalize_process_grid(global_device_count, process_grid)

    return DistributedRuntimePlan(
        process_count=process_count,
        process_index=process_index,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        process_grid=process_grid,
    )
