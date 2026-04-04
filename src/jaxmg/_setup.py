import importlib
import pathlib
import ctypes
import warnings
import sys
import os
import re
from dataclasses import dataclass

import jax
import jax.extend

from ._cusolvermp_targets import cusolvermp_setup_targets
from .utils import JaxMgWarning

_lib_dir = os.path.dirname(__file__)
_initialized = False

_CUDA_TARGETS = {
    "mg": {
        "SPMD": {
            "cyclic_mg": ("libcyclic.so", "CyclicMgFFI"),
            "potrs_mg": ("libpotrs.so", "PotrsMgFFI"),
            "potri_mg": ("libpotri.so", "PotriMgFFI"),
            "syevd_mg": ("libsyevd.so", "SyevdMgFFI"),
            "syevd_no_V_mg": ("libsyevd_no_V.so", "SyevdMgFFI"),
        },
        "MPMD": {
            "potrs_mg": ("libpotrs_mp.so", "PotrsMgMpFFI"),
            "potri_mg": ("libpotri_mp.so", "PotriMgMpFFI"),
            "syevd_mg": ("libsyevd_mp.so", "SyevdMgMpFFI"),
            "syevd_no_V_mg": ("libsyevd_no_V_mp.so", "SyevdNoVMgMpFFI"),
        },
    },
    "mp": cusolvermp_setup_targets(),
}


@dataclass(frozen=True)
class _CudaBackendConfig:
    backend_definition: "_BackendFamilyDefinition"
    bin_dir: str
    mode: str
    n_devices_per_node: int


@dataclass(frozen=True)
class _BackendFamilyDefinition:
    family: str
    cuda_targets: dict[str, dict[str, tuple[str, str]]]


def _resolve_backend_family_definition():
    backend_family = os.environ.get("JAXMG_BACKEND_FAMILY", "mg").strip().lower()
    supported_families = ", ".join(sorted(_CUDA_TARGETS))
    try:
        cuda_targets = _CUDA_TARGETS[backend_family]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported JAXMg backend family: {backend_family}. "
            f"Current supported values: {supported_families}"
        ) from exc
    return _BackendFamilyDefinition(family=backend_family, cuda_targets=cuda_targets)


def _resolve_cuda_bin_dir():
    backend = jax.extend.backend.get_backend()
    m = re.search(r"cuda[^0-9]*([0-9]+(?:\.[0-9]+)*)", backend.platform_version, re.I)
    if m:
        cuda_major = m.group(1)[:2]
    else:
        raise OSError("Unable to parse CUDA version")
    return f"cu{cuda_major}"


def _resolve_runtime_mode():
    if not jax.distributed.is_initialized():
        n_devices_per_node = jax.local_device_count()
        mode = "SPMD"
        return mode, n_devices_per_node

    if "JAXMG_NUMBER_OF_DEVICES" in os.environ:
        n_devices_per_node = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
        n_machines = jax.device_count() // n_devices_per_node
        warnings.warn(
            f"Running in MPMD mode, with JAXMG_NUMBER_OF_DEVICES={n_devices_per_node}."
            f"JAXMg is running on {n_machines} machines and {n_devices_per_node} devices per node."
            "If this configuation is incorrect, the code will hang or error.",
            JaxMgWarning,
            stacklevel=5,
        )
    else:
        n_devices_per_node = jax.device_count()
        warnings.warn(
            f"Running in MPMD mode with {n_devices_per_node} devices. "
            "By default, we assume that computation is running in a single node. "
            "To run JAXMg in a setting with multiple nodes, manually set JAXMG_NUMBER_OF_DEVICES to the number of devices per node"
            " (see https://flatironinstitute.github.io/jaxmg/examples/spmd_mpmd/ for more details)",
            JaxMgWarning,
            stacklevel=5,
        )
    mode = "MPMD"
    return mode, n_devices_per_node


if not sys.platform.startswith("linux"):
    warnings.warn(
        f"Unsupported platform {sys.platform}, only Linux is supported. Non-Linux only works for docs.",
        JaxMgWarning,
        stacklevel=2,
    )


def _load(module, libraries):
    env_candidates = []
    if module == "cusolver":
        env_candidates.extend(
            [
                os.environ.get("JAXMG_CUSOLVERMG_LIB"),
                os.environ.get("CUSOLVERMG_LIB"),
            ]
        )
    elif module == "cusolvermp":
        env_candidates.extend(
            [
                os.environ.get("JAXMG_CUSOLVERMP_LIB"),
                os.environ.get("CUSOLVERMP_LIB"),
            ]
        )
    elif module == "nccl":
        env_candidates.extend(
            [
                os.environ.get("JAXMG_NCCL_LIB"),
                os.environ.get("NCCL_LIB"),
            ]
        )

    try:
        m = importlib.import_module(f"nvidia.{module}")
    except ImportError:
        m = None

    last_error = None
    for lib in libraries:
        candidates = [candidate for candidate in env_candidates if candidate]
        candidates.append(lib)

        if m is not None:
            candidates.append(str(pathlib.Path(m.__path__[0]) / "lib" / lib))

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidates.append(str(pathlib.Path(conda_prefix) / "lib" / lib))

        for candidate in candidates:
            try:
                ctypes.cdll.LoadLibrary(candidate)
                return
            except OSError as e:
                last_error = e

    raise OSError(
        f"Unable to load CUDA library {libraries}, make sure you have a version of JAX that is "
        "GPU compatible: jax[cuda12], jax[cuda12-local] (>=0.6.2) or jax[cuda13], jax[cuda13-local] (>=0.7.2)."
        "This is guaranteed if you install JAXMg as: jaxmg[cuda12], jaxmg[cuda12-local], jaxmg[cuda13] or jaxmg[cuda13-local]"
    ) from last_error


def _load_backend_dependencies(backend_family: str):
    if backend_family == "mg":
        _load("cusolver", ["libcusolverMg.so.11"])
        try:
            _load("cu13", ["libcusolverMg.so.12"])
        except OSError:
            pass
        return

    if backend_family == "mp":
        if os.environ.get("JAXMG_ENABLE_REAL_CUSOLVERMP", "").strip() == "1":
            _load("cusolvermp", ["libcusolverMp.so"])
            _load("nccl", ["libnccl.so.2", "libnccl.so"])
            return
        if os.environ.get("JAXMG_ENABLE_MP_STUB", "").strip() == "1":
            return
        raise NotImplementedError(
            "JAXMG_BACKEND_FAMILY=mp is recognized but not implemented yet. "
            "Set JAXMG_ENABLE_MP_STUB=1 to allow experimental stub registration "
            "or JAXMG_ENABLE_REAL_CUSOLVERMP=1 to require real cuSOLVERMp/NCCL dependencies."
        )

    raise ValueError(f"Unsupported JAXMg backend family: {backend_family}")


def _initialize_cuda_backend():
    config = _resolve_cuda_backend_config()

    _load_backend_dependencies(config.backend_definition.family)

    jax.config.update("jax_enable_x64", True)

    # set if not set already
    os.environ.setdefault("JAXMG_NUMBER_OF_DEVICES", str(config.n_devices_per_node))

    _register_cuda_targets(config.bin_dir, config.backend_definition, config.mode)


def _resolve_cuda_backend_config():
    backend_definition = _resolve_backend_family_definition()
    bin_dir = _resolve_cuda_bin_dir()
    mode, n_devices_per_node = _resolve_runtime_mode()
    return _CudaBackendConfig(
        backend_definition=backend_definition,
        bin_dir=bin_dir,
        mode=mode,
        n_devices_per_node=n_devices_per_node,
    )


def _initialize():
    if any("gpu" == d.platform for d in jax.devices()):
        _initialize_cuda_backend()

    else:
        warnings.warn(
            "No GPUs found, only use this mode for testing or generating documentation.",
            JaxMgWarning,
            stacklevel=4,  # _initialize -> ensure_init_jaxmg_backend -> public fn -> user code
        )
        os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jax.device_count())


def ensure_init_jaxmg_backend():
    global _initialized
    if _initialized:
        return
    _initialized = True
    _initialize()


def _resolve_cuda_targets(backend_definition: _BackendFamilyDefinition, mode: str):
    try:
        return backend_definition.cuda_targets[mode]
    except KeyError as exc:
        raise ValueError(
            "Unsupported CUDA target configuration for "
            f"backend_family={backend_definition.family!r}, mode={mode!r}"
        ) from exc


def _register_cuda_targets(bin_dir: str, backend_definition: _BackendFamilyDefinition, mode: str):
    targets = _resolve_cuda_targets(backend_definition, mode)
    for target_name, (lib_name, symbol_name) in targets.items():
        library = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/{lib_name}"))
        capsule = jax.ffi.pycapsule(getattr(library, symbol_name))
        jax.ffi.register_ffi_target(target_name, capsule, platform="CUDA")
