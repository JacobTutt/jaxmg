from pathlib import Path

import jax
import pytest

from mpmd_helper import run_mpmd_test

pytestmark = pytest.mark.mpmd

HERE = Path(__file__).parent
MP_TEST = HERE / "run_potrs_cusolvermp.py"

platforms = set(d.platform for d in jax.devices())
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)


def _cases():
    return ["diag", "dense_spd", "diag_row_sharded", "dense_spd_row_sharded"]


@pytest.mark.parametrize("name", _cases())
@pytest.mark.parametrize("dtype_name", ["float32"])
def test_potrs_cusolvermp_diag_multirank(name, dtype_name):
    gpu_count = jax.device_count()
    if gpu_count != 2:
        pytest.skip(f"Need exactly 2 visible GPUs for this regression (have {gpu_count})")

    run_mpmd_test(MP_TEST, requested_procs=2, name=name, dtype_name=dtype_name)
