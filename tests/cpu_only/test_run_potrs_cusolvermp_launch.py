import os

from tests.mpmd.run_potrs_cusolvermp import LaunchConfig, _resolve_launch_config


def test_resolve_launch_config_from_argv(monkeypatch):
    monkeypatch.delenv("SLURM_LOCALID", raising=False)
    cfg = _resolve_launch_config(
        ["run_potrs_cusolvermp.py", "127.0.0.1:1234", "1", "2", "diag", "float32"]
    )
    assert cfg == LaunchConfig(
        coordinator_address="127.0.0.1:1234",
        process_id=1,
        num_processes=2,
        local_device_id=1,
        task_name="diag",
        task_dtype_name="float32",
    )


def test_resolve_launch_config_prefers_slurm_localid(monkeypatch):
    monkeypatch.setenv("SLURM_LOCALID", "0")
    cfg = _resolve_launch_config(
        ["run_potrs_cusolvermp.py", "127.0.0.1:1234", "1", "2", "diag", "float32"]
    )
    assert cfg.local_device_id == 0


def test_resolve_launch_config_from_env(monkeypatch):
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "nid0:12395")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("JAXMG_MPTEST_NAME", "diag_row_sharded")
    monkeypatch.setenv("JAXMG_MPTEST_DTYPE", "float64")
    cfg = _resolve_launch_config(["run_potrs_cusolvermp.py"])
    assert cfg == LaunchConfig(
        coordinator_address="nid0:12395",
        process_id=3,
        num_processes=8,
        local_device_id=1,
        task_name="diag_row_sharded",
        task_dtype_name="float64",
    )
