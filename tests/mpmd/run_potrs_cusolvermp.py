import json
import os
import socket
import sys
import traceback
from typing import Callable, Dict, List

import jax
import jax.distributed
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[proc_id],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

os.environ["JAXMG_BACKEND_FAMILY"] = "mp"
os.environ["JAXMG_ENABLE_REAL_CUSOLVERMP"] = "1"
os.environ.pop("JAXMG_ENABLE_MP_STUB", None)

mesh = jax.make_mesh((jax.device_count(),), ("x",))

from jaxmg import potrs


def _println(prefix: str, payload: dict):
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


def _solve_diag(dtype):
    a = jnp.diag(jnp.arange(1, 5, dtype=dtype))
    b = jnp.ones((4,), dtype=dtype)
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
    expected = jnp.asarray([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=dtype)
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
        "out": [float(x) for x in jnp.asarray(out)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _build_registry() -> Dict[str, Callable]:
    return {"diag": _solve_diag}


def main(argv: List[str]):
    task_name = argv[4]
    task_dtype_name = argv[5]
    registry = _build_registry()

    dtypes = {
        "float32": jnp.float32,
        "float64": jnp.float64,
    }
    dt = dtypes[task_dtype_name]
    fn = registry[task_name]

    _println(
        "MPTEST_DISCOVER",
        {
            "proc": proc_id,
            "available": sorted(registry.keys()),
            "selected": [task_name],
            "params": {"dtype": [task_dtype_name]},
        },
    )

    try:
        payload = fn(dt)
        _println(
            "MPTEST_RESULT",
            {
                "proc": proc_id,
                "name": task_name,
                "status": "ok",
                "params": {"dtype": task_dtype_name},
                "payload": payload,
            },
        )
        _println("MPTEST_SUMMARY", {"proc": proc_id, "ok": 1, "fail": 0, "total": 1})
        return 0
    except Exception:
        _println(
            "MPTEST_RESULT",
            {
                "proc": proc_id,
                "name": task_name,
                "status": "fail",
                "params": {"dtype": task_dtype_name},
                "traceback": traceback.format_exc(limit=40),
            },
        )
        _println("MPTEST_SUMMARY", {"proc": proc_id, "ok": 0, "fail": 1, "total": 1})
        return 1


raise SystemExit(main(sys.argv))
