import pytest

from jaxmg._setup import (
    _BackendFamilyDefinition,
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
    assert definition.cuda_targets == {"SPMD": {}, "MPMD": {}}


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


def test_resolve_cuda_targets_rejects_missing_mode(monkeypatch):
    monkeypatch.setenv("JAXMG_BACKEND_FAMILY", "mg")

    definition = _resolve_backend_family_definition()

    with pytest.raises(ValueError, match="mode='INVALID'"):
        _resolve_cuda_targets(definition, "INVALID")
