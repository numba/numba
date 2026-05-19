#!/usr/bin/env python
"""Tests for matrix_spec.

Run: python buildscripts/github/test_matrix_evaluate.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import (  # noqa: E402
    PLATFORMS,
    CONDA_BUILD_MATRIX,
    WHEEL_BUILD_MATRIX,
    evaluate,
    _python_tag,
    _canonical_version,
)


def test_python_tag():
    # major.minor -> cpXY
    assert _python_tag("3.10") == "cp310"
    assert _python_tag("3.14") == "cp314"
    # free-threaded suffix preserved
    assert _python_tag("3.14t") == "cp314t"
    # patch component stripped (with/without 't')
    assert _python_tag("3.14.3") == "cp314"
    assert _python_tag("3.14.3t") == "cp314t"


def test_canonical_version():
    # already-canonical input is returned unchanged
    assert _canonical_version("3.10") == "3.10"
    assert _canonical_version("3.14") == "3.14"
    # free-threaded suffix dropped
    assert _canonical_version("3.14t") == "3.14"
    # patch component dropped (with/without 't')
    assert _canonical_version("3.14.3") == "3.14"
    assert _canonical_version("3.14.3t") == "3.14"


def test_wheel_has_free_threaded():
    py_versions = {r["python-version"] for r in WHEEL_BUILD_MATRIX}
    # wheel matrix includes the free-threaded build
    assert "3.14t" in py_versions
    # conda matrix does not (no upstream FT conda package)
    assert "3.14t" not in {r["python-version"] for r in CONDA_BUILD_MATRIX}


def test_wheel_314t_fields():
    entry = next(
        r for r in WHEEL_BUILD_MATRIX if r["python-version"] == "3.14t"
    )
    # python_tag keeps 't'; python_canonical strips it
    assert entry["python_tag"] == "cp314t"
    assert entry["python_canonical"] == "3.14"


def test_conda_has_python_canonical():
    # field present on every conda build row
    assert all("python_canonical" in r for r in CONDA_BUILD_MATRIX)
    # and its value is MAJOR.MINOR with both parts numeric
    for r in CONDA_BUILD_MATRIX:
        parts = r["python_canonical"].split(".")
        assert len(parts) == 2, r
        assert all(p.isdigit() for p in parts), r


def test_eval_pull_request():
    # default pull_request returns full matrix, tagged with platform
    for pkg in ("conda", "wheel"):
        expected = (
            CONDA_BUILD_MATRIX if pkg == "conda" else WHEEL_BUILD_MATRIX
        )
        for p in PLATFORMS:
            build, _ = evaluate(pkg, "pull_request", None, p)
            assert len(build) == len(expected), (pkg, p)
            assert all(r["platform"] == p for r in build), (pkg, p)


def test_eval_label():
    # build_numba_wheel selects wheel only: full wheel, empty conda
    build, _ = evaluate(
        "wheel", "pull_request", ["build_numba_wheel"], "linux-64",
    )
    assert len(build) == len(WHEEL_BUILD_MATRIX)
    build, _ = evaluate(
        "conda", "pull_request", ["build_numba_wheel"], "linux-64",
    )
    assert build == []
    # unrelated label: selects full matrices - wheel+conda
    build, _ = evaluate(
        "wheel", "pull_request", ["foo"], "linux-64",
    )
    assert len(build) == len(WHEEL_BUILD_MATRIX)
    build, _ = evaluate(
        "conda", "pull_request", ["foo"], "linux-64",
    )
    assert len(build) == len(CONDA_BUILD_MATRIX)


def test_eval_unknown_event_returns_empty():
    # unknown event yields empty build/test matrices
    for pkg in ("conda", "wheel"):
        build, test = evaluate(pkg, "foo", None, "linux-64")
        assert build == [] and test == []


def test_eval_dispatch_filter_python():
    # workflow_dispatch python_version input narrows matrix to that row
    build, _ = evaluate(
        "wheel", "workflow_dispatch", None,
        "linux-64", json.dumps({"python_version": "3.14t"}),
    )
    assert len(build) == 1
    assert build[0]["python_tag"] == "cp314t"
    assert build[0]["python_canonical"] == "3.14"


def test_eval_dispatch_filter_numpy():
    # numpy inputs filter build and test matrices independently
    build, test = evaluate(
        "wheel", "workflow_dispatch", None,
        "linux-64",
        json.dumps({
            "numpy_build_version": "2.0.2",
            "numpy_test_version": "2.0",
        }),
    )
    assert all(r["numpy_build"] == "2.0.2" for r in build)
    assert all(r["numpy_test"] == "2.0" for r in test)


if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        else:
            print(f"OK   {t.__name__}")
    print(f"\n{len(tests)} tests, {failed} failed")
    sys.exit(1 if failed else 0)
