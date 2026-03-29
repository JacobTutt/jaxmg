import os
from functools import partial
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._cyclic_1d import calculate_padding, pad_rows
from ._setup import ensure_init_jaxmg_backend


def potrs(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P] | P,
    return_status: bool = False,
    return_logdet: bool = False,
    pad=True,
) -> Union[Array, Tuple[Array, Array], Tuple[Array, int], Tuple[Array, Array, int]]:
    """Solve the linear system A x = B using the multi-GPU potrs native kernel.

    Prepares inputs for the native ``potrs_mg`` kernel and executes it via
    ``jax.ffi.ffi_call`` under ``jax.jit`` and ``jax.shard_map``. Handles
    per-device padding driven by ``T_A`` and returns the solution (and
    optionally a host-side solver status).

    Tip:
        If the shards of the matrix cannot be padded with tiles of size `T_A`
        (``N / num_gpus % T_A != 0``) we have to add padding to fit the last tile.
        This requires copying the matrix, which we want to avoid at all costs for
        large ``N``. Make sure you pick ``T_A`` large enough (>=128) and such that it
        can evenly cover the shards. In principle, increasing ``T_A`` will increase
        performance at the cost of memory, but depending on ``N``, the performance
          will saturate.

    Args:
        a (Array): 2D, symmetric matrix representing the coefficient matrix.
            Expected to be sharded across the mesh along the first (row) axis
            using a single ``PartitionSpec``: ``P(<axis_name>, None)``.
        b (Array): 2D right-hand side. Expected to be replicated across
            devices with ``PartitionSpec`` ``P(None, None)`` or ``P(None)``.
        T_A (int): Tile width used by the native solver. Each
            local shard length must be a multiple of ``T_A``. If the user provides a
            ``T_A`` that is incompatible with the shard size we pad the matrix
            accordingly. For small tile sizes (``T_A``< 128), the solver can
            be extremely slow, so ensure that ``T_A`` is large enough. In principle,
            the larger ``T_A`` the faster the solver runs. See https://arxiv.org/abs/2601.14466
            for more details.
        mesh (Mesh): JAX Mesh object used for ``jax.shard_map``.
        in_specs (PartitionSpec or tuple/list[PartitionSpec]): PartitionSpec
            describing the input sharding (row sharding). May be provided as a
            single ``PartitionSpec`` or a single-element container containing one.
        return_status (bool, optional): If True append the host-replicated
            native solver status to the return value. Default is False.
        return_logdet (bool, optional): If True also return ``log|A|`` computed
            from the Cholesky factor already produced inside ``potrs``. The
            factor stays internal to the jitted wrapper; only the scalar logdet
            is returned. Default is False.
        pad (bool, optional): If True (default) apply per-device padding to
            ``a`` so each local shard length is compatible with ``T_A``; if
            False the caller must ensure shapes already match the kernel's
            requirements.

    Returns:
        One of:
            - ``x``
            - ``(x, status)``
            - ``(x, logdet)``
            - ``(x, logdet, status)``
        where ``x`` is the solution replicated across devices and ``logdet``
        is the scalar ``log|A|``.

    Raises:
        AssertionError: If ``a`` or ``b`` are not the correct shape, or if their shapes
            are incompatible.
        ValueError: If ``in_specs`` is not a 1-element sequence or if the provided
            ``PartitionSpec`` objects do not match the required patterns
            (``P(<axis_name>, None)`` for ``a``).
    Notes:
        - The FFI call may donate the ``a`` buffer (``donate_argnums=0``) for
          zero-copy interaction with the native library.
        - If the native solver fails the returned solution may contain NaNs and
          ``status`` will be non-zero.
    """
    ensure_init_jaxmg_backend()

    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])

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
    if (in_specs._partitions[0] == None) or (in_specs._partitions[1] != None):
        raise ValueError(
            "A must be sharded along the rows with PartitionSpec P(str, None)."
        )

    assert a.shape[1] == b.shape[0], "A and b must have the same number of columns."
    assert a.ndim == 2, "a must be a 2D array."
    assert b.ndim <= 2, "b must be a 1D or 2D array."
    # ensure b is always 2D
    if b.ndim == 1:
        b = jnp.expand_dims(b, axis=1)

    N_rows, N = a.shape
    axis_name = in_specs._partitions[0]

    shard_size = N_rows // ndev

    # Keep b in column-major layout
    input_layouts = ((0, 1), (1, 0))
    output_layouts = ((0, 1), (1, 0), (0,))

    padding = calculate_padding(shard_size, T_A)

    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                N_rows == N + ndev * padding
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {N_rows}"
        # Identity padding
        pad_fn = lambda _a: _a
        padding = 0
    else:
        # Make padding fns
        pad_fn = jax.shard_map(
            partial(pad_rows, padding=padding),
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=True,
        )

    out_type = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    out_type_logdet = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )

    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "potrs_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 0, 1: 1},
        ),
        T_A=T_A,
    )
    ffi_fn_logdet = partial(
        jax.ffi.ffi_call(
            "potrs_logdet_mg",
            out_type_logdet,
            input_layouts=input_layouts,
            output_layouts=((0, 1), (1, 0), (0,), (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        T_A=T_A,
    )

    # Jit with donate_argnums=0 is crucial for buffer sharing
    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(axis_name, None), P(None, None)),
        out_specs=(P(axis_name, None), P(None, None), P(None)),
        check_vma=False,
    )
    def impl(_a, _b):
        _a = _a.conj()
        _out_a, _out_b, _status = ffi_fn(_a, _b)
        return _out_a, _out_b, _status

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(axis_name, None), P(None, None)),
        out_specs=(P(None, None), P(None), P(None)),
        check_vma=False,
    )
    def impl_with_native_logdet(_a, _b):
        _a = _a.conj()
        _out_a, _out_b, _logdet, _status = ffi_fn_logdet(_a, _b)
        return _out_b, _logdet, _status

    def fn(_a, _b):
        _a = pad_fn(_a)
        if return_logdet:
            _out_b, _logdet, _status = impl_with_native_logdet(_a, _b)
            return _out_b, _logdet, _status
        _out_a, _out_b, _status = impl(_a, _b)
        return _out_b, _status

    if return_logdet:
        out, logdet, status = fn(a, b)
        if return_status:
            return out, logdet[0], status[0]
        return out, logdet[0]

    out, status = fn(a, b)
    if return_status:
        return out, status[0]
    return out


def potrs_shardmap_ctx(
    a: Array,
    b: Array,
    T_A: int,
    return_logdet: bool = False,
    pad=True,
) -> Union[Tuple[Array, Array], Tuple[Array, Array, Array]]:
    """Solve A x = B by invoking the native multi-GPU potrs kernel without shard_map.

    This helper is a lightweight, lower-level variant of :func:`jaxmg.potrs` intended
    for contexts where the input ``a`` is already laid out and sharded at the
    application level (for example when running inside a custom
    ``shard_map``/pjit-managed context). It performs the same padding logic
    driven by ``T_A`` and directly calls the native ``potrs_mg`` FFI targets
    via ``jax.ffi.ffi_call`` instead of constructing an additional ``shard_map``
    wrapper.

    Tip:
        If the shards of the matrix cannot be padded with tiles of size `T_A`
        (``N / num_gpus % T_A != 0``) we have to add padding to fit the last tile.
        This requires copying the matrix, which we want to avoid at all costs for
        large ``N``. Make sure you pick ``T_A`` large enough (>=128) and such that it
        can evenly cover the shards. In principle, increasing ``T_A`` will increase
        performance at the cost of memory, but depending on ``N``, the performance
          will saturate.

    Args:
        a (Array): 2D coefficient matrix of shape ``(N_rows // ndev, N)``. Must be
            symmetric for correct solver behavior.
        b (Array): 2D right-hand side. Its first dimension must equal the
            number of columns of ``a`` (i.e. ``a.shape[1] == b.shape[0]``).
        T_A (int): Tile width used by the native solver. Each
            local shard length must be a multiple of ``T_A``. If the user provides a
            ``T_A`` that is incompatible with the shard size we pad the matrix
            accordingly. For small tile sizes (``T_A``< 128), the solver can
            be extremely slow, so ensure that ``T_A`` is large enough. In principle,
            the larger ``T_A`` the faster the solver runs.
        pad (bool, optional): If True (default) apply per-device padding to
            ``a`` so each local shard length is compatible with ``T_A``. If
            False the caller must ensure shapes already meet the kernel's
            requirements.

    Returns:
        tuple: ``(x, status)`` or ``(x, logdet, status)`` where ``x`` is the
            solver result (same shape as ``b``), ``logdet`` is a float64 device
            scalar with ``log|A|``, and ``status`` is the int32 status value
            returned by the native kernel.

    Raises:
        AssertionError: If input arrays are not 2D or their shapes are
            incompatible.

    Notes:
        - This function does not perform sharding via ``jax.shard_map`` and
          therefore must be called only in a shard_map context.
        - Because it does not use ``donate_argnums``, the input buffers are
          not donated to the FFI call (no zero-copy donation semantics).
    """
    ensure_init_jaxmg_backend()
    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    assert a.shape[1] == b.shape[0], "A and b must have the same number of rows."
    assert a.ndim == 2, "a must be a 2D array."
    assert b.ndim == 2, "b must be a 2D array."
    shard_size, N = a.shape

    # Keep b in column-major layout
    input_layouts = ((0, 1), (1, 0))
    output_layouts = ((0, 1), (1, 0), (0,))

    padding = calculate_padding(shard_size, T_A)

    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                shard_size == (N + ndev * padding) // ndev
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {shard_size}"
        # Identity padding
        pad_fn = lambda _a: _a
        padding = 0

    else:
        # Make padding fns
        pad_fn = partial(pad_rows, padding=padding)

    out_type = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    out_type_logdet = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )

    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "potrs_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 0, 1: 1},
        ),
        T_A=T_A,
    )
    ffi_fn_logdet = partial(
        jax.ffi.ffi_call(
            "potrs_logdet_mg",
            out_type_logdet,
            input_layouts=input_layouts,
            output_layouts=((0, 1), (1, 0), (0,), (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        T_A=T_A,
    )

    # Jit with donate_argnums=0 is crucial for buffer sharing
    def impl(_a, _b):
        _a = _a.conj()
        _out_a, _out_b, _status = ffi_fn(_a, _b)
        return _out_b, _status

    def impl_with_native_logdet(_a, _b):
        _a = _a.conj()
        _out_a, _out_b, _logdet, _status = ffi_fn_logdet(_a, _b)
        return _out_b, _logdet, _status

    def fn(_a, _b):
        _a = pad_fn(_a)
        if return_logdet:
            _out, _logdet, _status = impl_with_native_logdet(_a, _b)
            return _out, _logdet, _status
        _out, _status = impl(_a, _b)
        return _out, _status

    return fn(a, b)
