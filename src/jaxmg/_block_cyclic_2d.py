import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockCyclicLayout:
    global_rows: int
    global_cols: int
    mb: int
    nb: int
    nprow: int
    npcol: int
    process_row: int
    process_col: int
    local_rows: int
    local_cols: int
    lld: int


def choose_process_grid(world_size: int, nprow: int | None = None, npcol: int | None = None) -> tuple[int, int]:
    """Return a near-square 2D process grid.

    If both dimensions are provided they must multiply to ``world_size``.
    If only one is provided the other is inferred.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, received {world_size}")

    if nprow is not None and nprow <= 0:
        raise ValueError(f"nprow must be positive, received {nprow}")
    if npcol is not None and npcol <= 0:
        raise ValueError(f"npcol must be positive, received {npcol}")

    if nprow is not None and npcol is not None:
        if nprow * npcol != world_size:
            raise ValueError(
                f"Invalid process grid ({nprow}, {npcol}) for world_size={world_size}"
            )
        return nprow, npcol

    if nprow is not None:
        if world_size % nprow != 0:
            raise ValueError(
                f"world_size={world_size} is not divisible by nprow={nprow}"
            )
        return nprow, world_size // nprow

    if npcol is not None:
        if world_size % npcol != 0:
            raise ValueError(
                f"world_size={world_size} is not divisible by npcol={npcol}"
            )
        return world_size // npcol, npcol

    root = math.isqrt(world_size)
    for rows in range(root, 0, -1):
        if world_size % rows == 0:
            return rows, world_size // rows
    raise AssertionError("unreachable")


def _validate_numroc_inputs(n: int, nb: int, iproc: int, isrcproc: int, nprocs: int) -> None:
    if n < 0:
        raise ValueError(f"n must be non-negative, received {n}")
    if nb <= 0:
        raise ValueError(f"nb must be positive, received {nb}")
    if nprocs <= 0:
        raise ValueError(f"nprocs must be positive, received {nprocs}")
    if not 0 <= iproc < nprocs:
        raise ValueError(f"iproc={iproc} must satisfy 0 <= iproc < {nprocs}")
    if not 0 <= isrcproc < nprocs:
        raise ValueError(f"isrcproc={isrcproc} must satisfy 0 <= isrcproc < {nprocs}")


def numroc(n: int, nb: int, iproc: int, isrcproc: int, nprocs: int) -> int:
    """Return the local extent of a block-cyclic distributed dimension.

    This follows the standard ScaLAPACK-style block-cyclic ownership model.
    """
    _validate_numroc_inputs(n, nb, iproc, isrcproc, nprocs)

    local = 0
    for block_start in range(0, n, nb):
        owner = ((block_start // nb) + isrcproc) % nprocs
        if owner != iproc:
            continue
        local += min(nb, n - block_start)
    return local


def local_matrix_shape(
    m: int,
    n: int,
    mb: int,
    nb: int,
    myprow: int,
    mypcol: int,
    nprow: int,
    npcol: int,
) -> tuple[int, int]:
    """Return the local matrix shape for a 2D block-cyclic distribution."""
    local_rows = numroc(m, mb, myprow, 0, nprow)
    local_cols = numroc(n, nb, mypcol, 0, npcol)
    return local_rows, local_cols


def owning_process_row(global_row: int, mb: int, nprow: int) -> int:
    """Return the process-row index that owns ``global_row``."""
    if global_row < 0:
        raise ValueError(f"global_row must be non-negative, received {global_row}")
    if mb <= 0:
        raise ValueError(f"mb must be positive, received {mb}")
    if nprow <= 0:
        raise ValueError(f"nprow must be positive, received {nprow}")
    return (global_row // mb) % nprow


def owning_process_col(global_col: int, nb: int, npcol: int) -> int:
    """Return the process-column index that owns ``global_col``."""
    if global_col < 0:
        raise ValueError(f"global_col must be non-negative, received {global_col}")
    if nb <= 0:
        raise ValueError(f"nb must be positive, received {nb}")
    if npcol <= 0:
        raise ValueError(f"npcol must be positive, received {npcol}")
    return (global_col // nb) % npcol


def local_leading_dimension(local_rows: int) -> int:
    """Return the leading dimension for a local column-major matrix buffer."""
    if local_rows < 0:
        raise ValueError(
            f"local_rows must be non-negative, received {local_rows}"
        )
    return max(1, local_rows)


def make_block_cyclic_layout(
    m: int,
    n: int,
    mb: int,
    nb: int,
    process_row: int,
    process_col: int,
    nprow: int,
    npcol: int,
) -> BlockCyclicLayout:
    """Describe the local block-cyclic storage owned by a single process."""
    local_rows, local_cols = local_matrix_shape(
        m=m,
        n=n,
        mb=mb,
        nb=nb,
        myprow=process_row,
        mypcol=process_col,
        nprow=nprow,
        npcol=npcol,
    )
    return BlockCyclicLayout(
        global_rows=m,
        global_cols=n,
        mb=mb,
        nb=nb,
        nprow=nprow,
        npcol=npcol,
        process_row=process_row,
        process_col=process_col,
        local_rows=local_rows,
        local_cols=local_cols,
        lld=local_leading_dimension(local_rows),
    )


def global_to_local_index(
    global_row: int,
    global_col: int,
    mb: int,
    nb: int,
    process_row: int,
    process_col: int,
    nprow: int,
    npcol: int,
) -> tuple[int, int] | None:
    """Map a global matrix entry to rank-local column-major coordinates.

    Returns ``None`` when the entry is not owned by the given process-grid
    coordinates.
    """
    owner_row = owning_process_row(global_row, mb, nprow)
    owner_col = owning_process_col(global_col, nb, npcol)
    if owner_row != process_row or owner_col != process_col:
        return None

    block_row = global_row // mb
    block_col = global_col // nb
    local_block_row = block_row // nprow
    local_block_col = block_col // npcol
    local_row = local_block_row * mb + (global_row % mb)
    local_col = local_block_col * nb + (global_col % nb)
    return local_row, local_col


def local_rhs_shape(
    n: int,
    mb: int,
    process_row: int,
    process_col: int,
    nprow: int,
    npcol: int,
) -> tuple[int, int]:
    """Return the local shape of a distributed single-RHS ``B`` matrix.

    Under a standard 2D block-cyclic descriptor for a global ``(N, 1)`` RHS,
    only one process-column owns the single global column.
    """
    return local_matrix_shape(
        m=n,
        n=1,
        mb=mb,
        nb=1,
        myprow=process_row,
        mypcol=process_col,
        nprow=nprow,
        npcol=npcol,
    )
