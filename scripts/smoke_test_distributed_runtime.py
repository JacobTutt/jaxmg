"""Smoke test for the JAX-side distributed solver grid assumptions.

This does not call cuSOLVERMp yet. It validates that:
- JAX distributed is initialized
- one process maps cleanly onto one logical solver rank
- the solver process grid metadata is derived consistently on every rank
"""

from __future__ import annotations

import json
import os

import jax

from jaxmg._distributed_solver import get_solver_process_grid, require_distributed_runtime


def _visible_local_device_ids() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return []
    return [int(part) for part in visible.split(",") if part.strip()]


def main() -> None:
    if not jax.distributed.is_initialized():
        coordinator_address = os.environ.get(
            "JAXMG_COORDINATOR_ADDRESS", "127.0.0.1:54321"
        )
        num_processes = int(os.environ.get("JAXMG_NUM_PROCESSES", "1"))
        process_id = int(os.environ.get("JAXMG_PROCESS_ID", "0"))
        local_device_ids = _visible_local_device_ids()
        if local_device_ids:
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=process_id,
                local_device_ids=local_device_ids,
                coordinator_bind_address=coordinator_address if process_id == 0 else None,
            )
        else:
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=process_id,
                coordinator_bind_address=coordinator_address if process_id == 0 else None,
            )

    require_distributed_runtime()
    grid = get_solver_process_grid()

    payload = {
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "local_device_count": int(jax.local_device_count()),
        "device_count": int(jax.device_count()),
        "nprow": int(grid.nprow),
        "npcol": int(grid.npcol),
        "process_row": int(grid.process_row),
        "process_col": int(grid.process_col),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
