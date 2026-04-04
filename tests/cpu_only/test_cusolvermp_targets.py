from jaxmg import cusolvermp_target_specs


def test_cusolvermp_target_specs_cover_current_placeholder_operations():
    specs = cusolvermp_target_specs()

    assert [spec.operation for spec in specs] == ["potrs", "syevd"]
    assert specs[0].ffi_target_name == "potrs_cusolvermp"
    assert specs[1].ffi_target_name == "syevd_cusolvermp"
