from dataclasses import dataclass
from typing import Tuple

from jax import Array


@dataclass(frozen=True)
class CuSolverMpPotrsContract:
    rhs_shape: Tuple[int, ...]
    rhs_ndim: int
    normalized_rhs_shape: Tuple[int, int] | None
    input_nrhs: int | None
    max_nrhs: int
    supports_multi_rhs: bool
    supported: bool
    failure_reason: str | None


def plan_potrs_cusolvermp_contract(b: Array) -> CuSolverMpPotrsContract:
    """Capture the current cuSOLVERMp potrs RHS contract."""
    rhs_shape = tuple(b.shape)
    rhs_ndim = b.ndim
    max_nrhs = 1

    if rhs_ndim == 1:
        return CuSolverMpPotrsContract(
            rhs_shape=rhs_shape,
            rhs_ndim=rhs_ndim,
            normalized_rhs_shape=(b.shape[0], 1),
            input_nrhs=1,
            max_nrhs=max_nrhs,
            supports_multi_rhs=False,
            supported=True,
            failure_reason=None,
        )

    if rhs_ndim == 2:
        input_nrhs = b.shape[1]
        supported = input_nrhs <= max_nrhs
        failure_reason = None
        if not supported:
            failure_reason = (
                "cuSOLVERMp potrs currently supports only NRHS=1, "
                f"but got B with shape {rhs_shape}."
            )
        return CuSolverMpPotrsContract(
            rhs_shape=rhs_shape,
            rhs_ndim=rhs_ndim,
            normalized_rhs_shape=rhs_shape,
            input_nrhs=input_nrhs,
            max_nrhs=max_nrhs,
            supports_multi_rhs=False,
            supported=supported,
            failure_reason=failure_reason,
        )

    return CuSolverMpPotrsContract(
        rhs_shape=rhs_shape,
        rhs_ndim=rhs_ndim,
        normalized_rhs_shape=None,
        input_nrhs=None,
        max_nrhs=max_nrhs,
        supports_multi_rhs=False,
        supported=False,
        failure_reason=(
            "cuSOLVERMp potrs expects b to be 1D or 2D with a single RHS column."
        ),
    )
