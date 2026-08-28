"""Every prism module must import from anywhere — no cwd/sys.path tricks."""

import importlib
import os
import pkgutil

import pytest

import prism

# Modules whose import is intentionally heavy but must still succeed on CPU
# without network: everything. __main__ modules are excluded (they execute).
_SKIP_SUFFIXES = ("__main__",)


def _walk():
    for m in pkgutil.walk_packages(prism.__path__, prefix="prism."):
        if m.name.endswith(_SKIP_SUFFIXES):
            continue
        yield m.name


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in list(os.environ):
        if var.startswith("PRISM_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # imports must not depend on the repo cwd


# Deps that only exist under optional extras — modules needing them are
# skipped (not failed) on a core-only install.
_EXTRA_DEPS = {"pandas", "sklearn", "scipy", "openpyxl", "vllm", "modal",
               "weave", "bert_score", "wandb"}


def _import_or_skip(name):
    try:
        importlib.import_module(name)
    except ImportError as e:
        missing = (e.name or "").split(".")[0]
        if missing in _EXTRA_DEPS:
            pytest.skip(f"{name} needs optional extra dep {missing!r}")
        raise


@pytest.mark.parametrize("name", sorted(_walk()))
def test_module_imports(name):
    _import_or_skip(name)


def test_no_sys_path_mutation():
    import sys
    before = list(sys.path)
    for name in sorted(_walk()):
        _import_or_skip(name)
    assert sys.path == before, "importing prism modules must not mutate sys.path"
