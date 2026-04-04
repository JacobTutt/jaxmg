import jax

from jaxmg import plan_cusolvermp_backend


def test_plan_cusolvermp_backend_captures_backend_contract():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))

    plan = plan_cusolvermp_backend(mesh)

    assert plan.backend_family == "mp"
    assert plan.distribution == "2d-block-cyclic"
    assert plan.local_layout == "column-major"
    assert plan.runtime.backend_family == "mp"
    assert plan.unsupported_operations == ("potri",)
    assert [target.operation for target in plan.targets] == ["potrs", "syevd"]
    assert all(target.implemented is False for target in plan.targets)


def test_plan_cusolvermp_backend_uses_explicit_process_grid():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))

    plan = plan_cusolvermp_backend(mesh, process_grid=(1, mesh.devices.size))

    assert plan.runtime.runtime.process_grid == (1, mesh.devices.size)
    assert plan.targets[0].ffi_target_name == "potrs_cusolvermp"
