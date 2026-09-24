"""The data roots come from the environment so a clone works on a machine with a different mount."""

import importlib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from mesospim_analysis import constants


def _reloaded() -> ModuleType:
    return importlib.reload(constants)


@pytest.fixture(autouse=True)
def _restore_defaults() -> Iterator[None]:
    yield
    importlib.reload(constants)


def test_defaults_apply_when_nothing_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MESOSPIM_DATA_ROOT", raising=False)
    monkeypatch.delenv("MESOSPIM_RESULTS_DIR", raising=False)
    module = _reloaded()
    assert module.DATA_ROOT == constants.DEFAULT_DATA_ROOT
    assert module.RESULTS_DIR == constants.DEFAULT_RESULTS_DIR


def test_the_environment_overrides_the_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESOSPIM_DATA_ROOT", "/home/james/mnt/MarcBusche/James/Mesospim")
    monkeypatch.setenv("MESOSPIM_RESULTS_DIR", "/scratch/afcorr")
    module = _reloaded()
    assert module.DATA_ROOT == Path("/home/james/mnt/MarcBusche/James/Mesospim")
    assert module.RESULTS_DIR == Path("/scratch/afcorr")


def test_an_empty_variable_falls_back_rather_than_becoming_the_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path("") is Path("."), which would silently scan the working directory for acquisitions."""
    monkeypatch.setenv("MESOSPIM_DATA_ROOT", "")
    module = _reloaded()
    assert module.DATA_ROOT == constants.DEFAULT_DATA_ROOT
