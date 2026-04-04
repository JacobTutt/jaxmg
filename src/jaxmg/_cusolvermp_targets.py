from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CuSolverMpTargetSpec:
    operation: str
    ffi_target_name: str
    library_name: str
    symbol_name: str


_CUSOLVERMP_TARGET_SPECS = (
    CuSolverMpTargetSpec(
        operation="potrs",
        ffi_target_name="potrs_cusolvermp",
        library_name="libpotrs_cusolvermp.so",
        symbol_name="PotrsCuSolverMpFFI",
    ),
    CuSolverMpTargetSpec(
        operation="syevd",
        ffi_target_name="syevd_cusolvermp",
        library_name="libsyevd_cusolvermp.so",
        symbol_name="SyevdCuSolverMpFFI",
    ),
)


def cusolvermp_target_specs() -> Tuple[CuSolverMpTargetSpec, ...]:
    return _CUSOLVERMP_TARGET_SPECS


def cusolvermp_setup_targets() -> Dict[str, Dict[str, Tuple[str, str]]]:
    targets = {
        spec.ffi_target_name: (spec.library_name, spec.symbol_name)
        for spec in _CUSOLVERMP_TARGET_SPECS
    }
    return {
        "SPMD": dict(targets),
        "MPMD": dict(targets),
    }
