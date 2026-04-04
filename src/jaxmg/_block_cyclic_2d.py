from math import isqrt
from typing import Tuple


def choose_process_grid(num_devices: int) -> Tuple[int, int]:
    """Choose a square-ish ``(nprow, npcol)`` grid whose product matches ``num_devices``."""
    if num_devices <= 0:
        raise ValueError("num_devices must be positive.")

    for nprow in range(isqrt(num_devices), 0, -1):
        if num_devices % nprow == 0:
            return (nprow, num_devices // nprow)

    raise AssertionError("Failed to construct a valid process grid.")


def normalize_block_shape(
    T_A: int, block_shape: Tuple[int, int] | None = None
) -> Tuple[int, int]:
    """Validate or infer the 2D block shape used for a future cyclic_2d layout."""
    if T_A <= 0:
        raise ValueError("T_A must be positive.")

    if block_shape is None:
        return (T_A, T_A)

    if len(block_shape) != 2:
        raise ValueError("block_shape must be a 2-tuple.")

    mb, nb = block_shape
    if mb <= 0 or nb <= 0:
        raise ValueError("block_shape entries must be positive.")
    return (mb, nb)


def normalize_process_grid(
    num_devices: int, process_grid: Tuple[int, int] | None = None
) -> Tuple[int, int]:
    """Validate or infer the process grid used for 2D block-cyclic ownership."""
    if process_grid is None:
        return choose_process_grid(num_devices)

    if len(process_grid) != 2:
        raise ValueError("process_grid must be a 2-tuple.")

    nprow, npcol = process_grid
    if nprow <= 0 or npcol <= 0:
        raise ValueError("process_grid entries must be positive.")
    if nprow * npcol != num_devices:
        raise ValueError(
            "process_grid must satisfy nprow * npcol == num_devices."
        )
    return (nprow, npcol)


def numroc(
    n: int,
    block_size: int,
    proc_coord: int,
    nprocs: int,
    src_proc: int = 0,
) -> int:
    """Return local rows/cols owned by a process under 2D block-cyclic distribution."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if nprocs <= 0:
        raise ValueError("nprocs must be positive.")
    if not 0 <= proc_coord < nprocs:
        raise ValueError("proc_coord must satisfy 0 <= proc_coord < nprocs.")
    if not 0 <= src_proc < nprocs:
        raise ValueError("src_proc must satisfy 0 <= src_proc < nprocs.")

    full_blocks, remainder = divmod(n, block_size)
    local_blocks = full_blocks // nprocs
    extra_blocks = full_blocks % nprocs
    owning_rank = (proc_coord - src_proc) % nprocs

    count = local_blocks * block_size
    if owning_rank < extra_blocks:
        count += block_size
    if owning_rank == extra_blocks:
        count += remainder
    return count


def local_block_cyclic_shape(
    global_shape: Tuple[int, int],
    block_shape: Tuple[int, int],
    process_grid: Tuple[int, int],
    process_coords: Tuple[int, int],
    src_process: Tuple[int, int] = (0, 0),
) -> Tuple[int, int]:
    """Return the rank-local matrix shape for a 2D block-cyclic global matrix."""
    nrows, ncols = global_shape
    mb, nb = block_shape
    nprow, npcol = process_grid
    prow, pcol = process_coords
    src_prow, src_pcol = src_process

    local_rows = numroc(nrows, mb, prow, nprow, src_proc=src_prow)
    local_cols = numroc(ncols, nb, pcol, npcol, src_proc=src_pcol)
    return (local_rows, local_cols)


def block_padding(size: int, block_size: int) -> int:
    """Return padding needed so ``size + padding`` is a multiple of ``block_size``."""
    if size < 0:
        raise ValueError("size must be non-negative.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return (-size) % block_size


def padded_block_shape(
    global_shape: Tuple[int, int], block_shape: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return ``(padded_shape, padding)`` for block-aligned 2D planning."""
    nrows, ncols = global_shape
    mb, nb = block_shape
    row_padding = block_padding(nrows, mb)
    col_padding = block_padding(ncols, nb)
    return ((nrows + row_padding, ncols + col_padding), (row_padding, col_padding))


def linear_rank_to_process_coords(
    rank: int, process_grid: Tuple[int, int]
) -> Tuple[int, int]:
    """Map a linear rank index into row-major ``(prow, pcol)`` coordinates."""
    nprow, npcol = process_grid
    num_devices = nprow * npcol
    if not 0 <= rank < num_devices:
        raise ValueError("rank must satisfy 0 <= rank < nprow * npcol.")
    return divmod(rank, npcol)


def process_grid_rank_order(process_grid: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
    """Return the row-major ``(prow, pcol)`` ordering for a process grid."""
    nprow, npcol = process_grid
    return tuple(
        linear_rank_to_process_coords(rank, process_grid)
        for rank in range(nprow * npcol)
    )
