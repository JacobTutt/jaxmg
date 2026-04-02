import os
from dataclasses import dataclass

import jax

from ._block_cyclic_2d import choose_process_grid


@dataclass(frozen=True)
class SolverProcessGrid:
    world_size: int
    nprow: int
    npcol: int
    process_index: int
    process_row: int
    process_col: int
    local_device_count: int


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def get_solver_process_grid(
    *,
    world_size: int | None = None,
    process_index: int | None = None,
    local_device_count: int | None = None,
    nprow: int | None = None,
    npcol: int | None = None,
) -> SolverProcessGrid:
    """Return the distributed solver grid metadata.

    Environment overrides:
    - ``JAXMG_SOLVER_NPROW``
    - ``JAXMG_SOLVER_NPCOL``
    """
    if world_size is None:
        world_size = jax.process_count()
    if process_index is None:
        process_index = jax.process_index()
    if local_device_count is None:
        local_device_count = jax.local_device_count()

    env_nprow = _env_int("JAXMG_SOLVER_NPROW")
    env_npcol = _env_int("JAXMG_SOLVER_NPCOL")
    if env_nprow is not None:
        nprow = env_nprow
    if env_npcol is not None:
        npcol = env_npcol

    nprow, npcol = choose_process_grid(world_size, nprow=nprow, npcol=npcol)

    if not 0 <= process_index < world_size:
        raise ValueError(
            f"process_index={process_index} must satisfy 0 <= process_index < {world_size}"
        )

    process_row = process_index // npcol
    process_col = process_index % npcol

    return SolverProcessGrid(
        world_size=world_size,
        nprow=nprow,
        npcol=npcol,
        process_index=process_index,
        process_row=process_row,
        process_col=process_col,
        local_device_count=local_device_count,
    )


def require_distributed_runtime() -> None:
    """Raise a clear error if the solver is invoked without JAX distributed."""
    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "The cuSOLVERMp migration path requires jax.distributed.initialize() "
            "with one process per GPU before calling solver routines."
        )
