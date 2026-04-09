import json
import os
import socket
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np

import jax
import jax.distributed
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


@dataclass(frozen=True)
class LaunchConfig:
    coordinator_address: str
    process_id: int
    num_processes: int
    local_device_id: int
    task_name: str
    task_dtype_name: str


def _resolve_task_local_device_id(default_id: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return default_id

    visible_ids = [part.strip() for part in visible.split(",") if part.strip()]
    if len(visible_ids) == 1:
        return 0

    return default_id


def _resolve_launch_config(argv: List[str]) -> LaunchConfig:
    if len(argv) >= 6:
        return LaunchConfig(
            coordinator_address=argv[1],
            process_id=int(argv[2]),
            num_processes=int(argv[3]),
            local_device_id=_resolve_task_local_device_id(
                int(os.environ.get("SLURM_LOCALID", argv[2]))
            ),
            task_name=argv[4],
            task_dtype_name=argv[5],
        )

    return LaunchConfig(
        coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
        process_id=int(os.environ["SLURM_PROCID"]),
        num_processes=int(os.environ["SLURM_NTASKS"]),
        local_device_id=_resolve_task_local_device_id(
            int(os.environ.get("SLURM_LOCALID", "0"))
        ),
        task_name=os.environ["JAXMG_MPTEST_NAME"],
        task_dtype_name=os.environ["JAXMG_MPTEST_DTYPE"],
    )


def _println(prefix: str, payload: dict):
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


def _initialize_distributed(config: LaunchConfig):
    jax.distributed.initialize(
        coordinator_address=config.coordinator_address,
        num_processes=config.num_processes,
        process_id=config.process_id,
        local_device_ids=[config.local_device_id],
        coordinator_bind_address=(
            config.coordinator_address if config.process_id == 0 else None
        ),
    )

    os.environ["JAXMG_BACKEND_FAMILY"] = "mp"
    os.environ["JAXMG_ENABLE_REAL_CUSOLVERMP"] = "1"
    os.environ.pop("JAXMG_ENABLE_MP_STUB", None)


def _solve_diag(dtype, mesh):
    host_a = jnp.diag(jnp.arange(1, 5, dtype=dtype))
    host_b = jnp.ones((4,), dtype=dtype)
    return _run_diag_solve(host_a, host_b, dtype, mesh)


def _make_dense_spd(dtype):
    lower = jnp.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.1, 2.5, 0.0, 0.0],
            [0.2, 0.3, 3.0, 0.0],
            [0.05, 0.15, 0.25, 3.5],
        ],
        dtype=dtype,
    )
    return lower @ lower.T.conj()


def _solve_dense_spd(dtype, mesh):
    host_a = _make_dense_spd(dtype)
    host_b = jnp.asarray([1.0, 0.5, -0.25, 0.75], dtype=dtype)
    return _run_diag_solve(host_a, host_b, dtype, mesh)


def _solve_diag_row_sharded(dtype, mesh):
    host_a = jnp.diag(jnp.arange(1, 5, dtype=dtype))
    host_b = jnp.ones((4,), dtype=dtype)
    a = jax.device_put(host_a, NamedSharding(mesh, P("x", None)))
    b = jax.device_put(host_b, NamedSharding(mesh, P(None)))
    return _run_diag_solve(a, b, dtype, mesh, host_a=host_a, host_b=host_b)


def _solve_dense_spd_row_sharded(dtype, mesh):
    host_a = _make_dense_spd(dtype)
    host_b = jnp.asarray([1.0, 0.5, -0.25, 0.75], dtype=dtype)
    a = jax.device_put(host_a, NamedSharding(mesh, P("x", None)))
    b = jax.device_put(host_b, NamedSharding(mesh, P(None)))
    return _run_diag_solve(a, b, dtype, mesh, host_a=host_a, host_b=host_b)


def _run_single_device_reference(host_a, host_b):
    device = jax.local_devices()[0]

    @jax.jit
    def _solve(lhs, rhs):
        return jnp.linalg.solve(lhs, rhs)

    with jax.default_device(device):
        ref_a = jnp.asarray(host_a)
        ref_b = jnp.asarray(host_b)
        return np.asarray(_solve(ref_a, ref_b))


def _serialize_value(x):
    if np.iscomplexobj(x):
        return {"real": float(np.real(x)), "imag": float(np.imag(x))}
    return float(np.real(x))


def _run_diag_solve(a, b, dtype, mesh, host_a=None, host_b=None):
    from jaxmg import potrs

    if host_a is None:
        host_a = np.asarray(jax.device_get(a))
    if host_b is None:
        host_b = np.asarray(jax.device_get(b))
    out, status = potrs(
        a,
        b,
        2,
        mesh=mesh,
        in_specs=(P("x", None),),
        process_grid=(jax.device_count(), 1),
        return_status=True,
    )
    out.block_until_ready()
    expected = _run_single_device_reference(host_a, host_b)
    assert int(status) == 0
    assert jnp.allclose(jnp.asarray(out), expected, atol=1e-6)

    return {
        "host": socket.gethostname(),
        "rank": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "device_count": jax.device_count(),
        "local_device_ids": [d.id for d in jax.local_devices()],
        "status": int(status),
        "out_shape": tuple(out.shape),
        "out": [_serialize_value(x) for x in np.asarray(out)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_localid": os.environ.get("SLURM_LOCALID"),
    }


def _build_registry(mesh) -> Dict[str, Callable]:
    return {
        "diag": lambda dtype: _solve_diag(dtype, mesh),
        "dense_spd": lambda dtype: _solve_dense_spd(dtype, mesh),
        "diag_row_sharded": lambda dtype: _solve_diag_row_sharded(dtype, mesh),
        "dense_spd_row_sharded": lambda dtype: _solve_dense_spd_row_sharded(dtype, mesh),
    }


def main(argv: List[str]):
    config = _resolve_launch_config(argv)
    _initialize_distributed(config)
    mesh = jax.make_mesh((jax.device_count(),), ("x",))
    registry = _build_registry(mesh)

    dtypes = {
        "float32": jnp.float32,
        "float64": jnp.float64,
        "complex128": jnp.complex128,
    }
    dt = dtypes[config.task_dtype_name]
    fn = registry[config.task_name]

    _println(
        "MPTEST_DISCOVER",
        {
            "proc": config.process_id,
            "available": sorted(registry.keys()),
            "selected": [config.task_name],
            "params": {"dtype": [config.task_dtype_name]},
        },
    )

    try:
        payload = fn(dt)
        _println(
            "MPTEST_RESULT",
            {
                "proc": config.process_id,
                "name": config.task_name,
                "status": "ok",
                "params": {"dtype": config.task_dtype_name},
                "payload": payload,
            },
        )
        _println(
            "MPTEST_SUMMARY",
            {"proc": config.process_id, "ok": 1, "fail": 0, "total": 1},
        )
        return 0
    except Exception:
        _println(
            "MPTEST_RESULT",
            {
                "proc": config.process_id,
                "name": config.task_name,
                "status": "fail",
                "params": {"dtype": config.task_dtype_name},
                "traceback": traceback.format_exc(limit=40),
            },
        )
        _println(
            "MPTEST_SUMMARY",
            {"proc": config.process_id, "ok": 0, "fail": 1, "total": 1},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
