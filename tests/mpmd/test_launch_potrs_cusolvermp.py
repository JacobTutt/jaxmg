import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import jax
import pytest

pytestmark = pytest.mark.mpmd

HERE = Path(__file__).parent
MP_TEST = HERE / "run_potrs_cusolvermp.py"

platforms = set(d.platform for d in jax.devices())
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)


def _cases():
    cases = ["diag", "dense_spd"]
    if os.environ.get("JAXMG_RUN_EXPERIMENTAL_ROW_SHARDED_MP", "").strip() == "1":
        cases.extend(["diag_row_sharded", "dense_spd_row_sharded"])
    return cases


def _submit_same_node_slurm_case(name: str, dtype_name: str) -> tuple[str, str]:
    run_root = Path("/home/u6n/jacobtutt.u6n/jaxmg_checkpoint_runs")
    repo_root = Path("/home/u6n/jacobtutt.u6n/jaxmg_multi_node")
    env_root = "/projects/u6n/jaxmg_multi_node/envs/jaxmg-isambard-20260404"
    stem = f"potrs_cusolvermp_pytracked_{name}_{dtype_name}"

    script = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=jmg_mp_tpotrs
        #SBATCH --partition=workq
        #SBATCH --nodes=1
        #SBATCH --ntasks=2
        #SBATCH --gpus-per-node=2
        #SBATCH --time=8:00
        #SBATCH --exclude=nid010460,nid010901,nid011087
        #SBATCH --output={run_root}/{stem}_%j.out
        #SBATCH --error={run_root}/{stem}_%j.err
        set -euo pipefail
        export JAX_COORDINATOR_ADDRESS="$(hostname):12419"
        export JAXMG_CUSOLVERMP_BOOTSTRAP_DIR={run_root}
        export JAXMG_CUSOLVERMP_BOOTSTRAP_TOKEN="{stem}_${{SLURM_JOB_ID}}_${{SLURM_STEP_ID:-0}}"
        cd {repo_root}
        srun --label bash -lc '
        set -euo pipefail
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate {env_root}
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export JAXMG_BACKEND_FAMILY=mp
        export JAXMG_ENABLE_REAL_CUSOLVERMP=1
        unset JAXMG_ENABLE_MP_STUB
        export JAX_COORDINATOR_ADDRESS="$JAX_COORDINATOR_ADDRESS"
        export JAXMG_CUSOLVERMP_BOOTSTRAP_DIR="$JAXMG_CUSOLVERMP_BOOTSTRAP_DIR"
        export JAXMG_CUSOLVERMP_BOOTSTRAP_TOKEN="$JAXMG_CUSOLVERMP_BOOTSTRAP_TOKEN"
        export JAXMG_MPTEST_NAME={name}
        export JAXMG_MPTEST_DTYPE={dtype_name}
        python -u {MP_TEST}
        '
        """
    )

    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as tmp:
        tmp.write(script)
        local_script = tmp.name

    remote_script = f"{run_root}/tracked_potrs_cusolvermp_{name}_{dtype_name}.sbatch"
    try:
        subprocess.run(
            ["scp", local_script, f"u6n.aip2.isambard:{remote_script}"],
            check=True,
            text=True,
            capture_output=True,
        )
        submit = subprocess.run(
            [
                "ssh",
                "u6n.aip2.isambard",
                f"cd {run_root} && sbatch {remote_script}",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    finally:
        Path(local_script).unlink(missing_ok=True)

    job_id = submit.stdout.strip().split()[-1]
    watcher = subprocess.run(
        [
            "/Users/jacobtutt/bin/watch-slurm-job",
            job_id,
            "--host",
            "u6n.aip2.isambard",
            "--interval",
            "10",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert watcher.returncode == 0, watcher.stdout + "\n" + watcher.stderr

    out_path = f"{run_root}/{stem}_{job_id}.out"
    err_path = f"{run_root}/{stem}_{job_id}.err"
    out = subprocess.run(
        ["ssh", "u6n.aip2.isambard", f"cat {out_path}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    err = subprocess.run(
        ["ssh", "u6n.aip2.isambard", f"cat {err_path}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return out, err


def _assert_runner_success(out: str, err: str, name: str, dtype_name: str):
    results = []
    summaries = []
    for line in out.splitlines():
        if line.startswith("0: MPTEST_RESULT ") or line.startswith("1: MPTEST_RESULT "):
            results.append(json.loads(line.split(" ", 2)[2]))
        elif line.startswith("0: MPTEST_SUMMARY ") or line.startswith("1: MPTEST_SUMMARY "):
            summaries.append(json.loads(line.split(" ", 2)[2]))

    assert len(results) == 2, f"Expected 2 MPTEST_RESULT lines.\nOUT:\n{out}\nERR:\n{err}"
    assert len(summaries) == 2, f"Expected 2 MPTEST_SUMMARY lines.\nOUT:\n{out}\nERR:\n{err}"
    failures = [r for r in results if r.get("status") != "ok"]
    assert not failures, f"Failures for {name}/{dtype_name}:\nOUT:\n{out}\nERR:\n{err}"


@pytest.mark.parametrize("name", _cases())
@pytest.mark.parametrize("dtype_name", ["float32"])
def test_potrs_cusolvermp_diag_multirank(name, dtype_name):
    out, err = _submit_same_node_slurm_case(name, dtype_name)
    _assert_runner_success(out, err, name, dtype_name)
