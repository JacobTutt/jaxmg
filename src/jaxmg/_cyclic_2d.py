from dataclasses import dataclass
from typing import List, Tuple

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._block_cyclic_2d import local_block_cyclic_shape, normalize_process_grid


@dataclass(frozen=True)
class Cyclic2DPlan:
    global_shape: Tuple[int, int]
    block_shape: Tuple[int, int]
    process_grid: Tuple[int, int]
    local_shapes: Tuple[Tuple[int, int], ...]


def cyclic_2d(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P],
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
) -> Cyclic2DPlan:
    """Planning-only preview of the future 2D block-cyclic layout helper.

    This intentionally does not invoke native code yet. It exists to establish
    the public migration seam while the live Mg solver path remains unchanged.
    """
    return plan_cyclic_2d_layout(
        a,
        T_A,
        mesh=mesh,
        in_specs=in_specs,
        pad=pad,
        process_grid=process_grid,
    )


def plan_cyclic_2d_layout(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P],
    pad: bool = True,
    process_grid: Tuple[int, int] | None = None,
) -> Cyclic2DPlan:
    """Plan the future 2D block-cyclic ownership without changing the live solver path."""
    del pad  # Padding policy is deferred until the real cyclic_2d implementation lands.

    if isinstance(in_specs, (list, tuple)):
        if len(in_specs) != 1:
            raise ValueError(
                "in_specs must be a single PartitionSpec or a 1-element list/tuple."
            )
        in_specs = in_specs[0]
    if not isinstance(in_specs, P):
        raise TypeError(
            "in_specs must be a PartitionSpec or a 1-element list/tuple containing one."
        )
    if a.ndim != 2:
        raise AssertionError("a must be a 2D array.")
    if (in_specs._partitions[1] is not None) or (in_specs._partitions[0] is None):
        raise ValueError(
            "A must be sharded along the rows with PartitionSpec P(str, None)."
        )
    if T_A <= 0:
        raise ValueError("T_A must be positive.")

    num_devices = mesh.devices.size
    process_grid = normalize_process_grid(num_devices, process_grid)
    nprow, npcol = process_grid
    block_shape = (T_A, T_A)

    local_shapes = tuple(
        local_block_cyclic_shape(a.shape, block_shape, process_grid, (prow, pcol))
        for prow in range(nprow)
        for pcol in range(npcol)
    )

    return Cyclic2DPlan(
        global_shape=a.shape,
        block_shape=block_shape,
        process_grid=process_grid,
        local_shapes=local_shapes,
    )
