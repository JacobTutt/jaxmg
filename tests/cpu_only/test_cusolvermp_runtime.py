import jax

from jaxmg import plan_cusolvermp_runtime


def test_plan_cusolvermp_runtime_captures_fixed_backend_assumptions():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))

    plan = plan_cusolvermp_runtime(mesh)

    assert plan.backend_family == "mp"
    assert plan.launch_model == "one-process-per-gpu"
    assert plan.requires_mpi is True
    assert plan.requires_nccl is True
    assert plan.supports_multi_node is True
    assert plan.supports_potri is False
    assert plan.supports_multi_rhs is False
    assert plan.max_nrhs == 1
    assert plan.runtime.global_device_count == mesh.devices.size
    assert plan.readiness.expected_process_count == mesh.devices.size
    assert plan.readiness.expected_local_device_count == 1


def test_plan_cusolvermp_runtime_reports_launch_readiness_for_single_device_session():
    mesh = jax.make_mesh((1,), ("x",))

    plan = plan_cusolvermp_runtime(mesh, process_grid=(1, 1))

    assert plan.readiness.launch_ready is True
    assert plan.readiness.issues == ()


def test_plan_cusolvermp_runtime_uses_explicit_process_grid():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))

    plan = plan_cusolvermp_runtime(mesh, process_grid=(1, mesh.devices.size))

    assert plan.runtime.process_grid == (1, mesh.devices.size)


def test_plan_cusolvermp_runtime_flags_multi_device_single_process_sessions():
    if jax.local_device_count() < 2:
        return

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))

    plan = plan_cusolvermp_runtime(mesh)

    assert plan.readiness.launch_ready is False
    assert any("one JAX process per GPU" in issue for issue in plan.readiness.issues)
    assert any("exactly one local GPU" in issue for issue in plan.readiness.issues)
