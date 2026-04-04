import pytest
from unittest import mock

from jaxmg._setup import (
    _BackendFamilyDefinition,
    _load_backend_dependencies,
    _resolve_backend_family_definition,
    _resolve_cuda_targets,
)


def test_resolve_backend_family_definition_defaults_to_mg(monkeypatch):
    monkeypatch.delenv("JAXMG_BACKEND_FAMILY", raising=False)

    definition = _resolve_backend_family_definition()

    assert isinstance(definition, _BackendFamilyDefinition)
    assert definition.family == "mg"
    assert "SPMD" in definition.cuda_targets
    assert "MPMD" in definition.cuda_targets


def test_resolve_backend_family_definition_recognizes_mp(monkeypatch):
    monkeypatch.setenv("JAXMG_BACKEND_FAMILY", "mp")

    definition = _resolve_backend_family_definition()

    assert definition.family == "mp"
    assert "potrs_cusolvermp" in definition.cuda_targets["SPMD"]
    assert "syevd_cusolvermp" in definition.cuda_targets["MPMD"]


def test_resolve_backend_family_definition_rejects_unknown(monkeypatch):
    monkeypatch.setenv("JAXMG_BACKEND_FAMILY", "unknown")

    with pytest.raises(ValueError, match="Current supported values: mg, mp"):
        _resolve_backend_family_definition()


def test_resolve_cuda_targets_uses_backend_definition(monkeypatch):
    monkeypatch.setenv("JAXMG_BACKEND_FAMILY", "mg")

    definition = _resolve_backend_family_definition()
    targets = _resolve_cuda_targets(definition, "SPMD")

    assert "potrs_mg" in targets
    assert targets["potrs_mg"] == ("libpotrs.so", "PotrsMgFFI")


def test_resolve_cuda_targets_exposes_mp_placeholder_targets(monkeypatch):
    monkeypatch.setenv("JAXMG_BACKEND_FAMILY", "mp")

    definition = _resolve_backend_family_definition()
    targets = _resolve_cuda_targets(definition, "SPMD")

    assert targets["potrs_cusolvermp"] == (
        "libpotrs_cusolvermp.so",
        "PotrsCuSolverMpFFI",
    )


def test_resolve_cuda_targets_rejects_missing_mode(monkeypatch):
    monkeypatch.setenv("JAXMG_BACKEND_FAMILY", "mg")

    definition = _resolve_backend_family_definition()

    with pytest.raises(ValueError, match="mode='INVALID'"):
        _resolve_cuda_targets(definition, "INVALID")


def test_load_backend_dependencies_requires_opt_in_for_mp_stub(monkeypatch):
    monkeypatch.delenv("JAXMG_ENABLE_MP_STUB", raising=False)

    with pytest.raises(NotImplementedError, match="JAXMG_ENABLE_MP_STUB=1"):
        _load_backend_dependencies("mp")


def test_load_backend_dependencies_allows_opted_in_mp_stub(monkeypatch):
    monkeypatch.setenv("JAXMG_ENABLE_MP_STUB", "1")

    _load_backend_dependencies("mp")


def test_load_backend_dependencies_allows_real_cusolvermp(monkeypatch):
    monkeypatch.setenv("JAXMG_ENABLE_REAL_CUSOLVERMP", "1")

    with mock.patch("jaxmg._setup._load") as load:
        _load_backend_dependencies("mp")

    load.assert_any_call("cusolvermp", ["libcusolverMp.so"])
    load.assert_any_call("nccl", ["libnccl.so.2", "libnccl.so"])
