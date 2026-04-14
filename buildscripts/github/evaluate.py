#!/usr/bin/env python
"""Declarative build/test matrices for GHA workflows.

Conda and wheel matrices are defined separately so either can
change independently.  Platform is added at evaluate time since
all platforms currently share the same version sets.

When run directly, reads GHA env vars and writes matrices to
$GITHUB_OUTPUT.
"""

import json
import os
from pathlib import Path

PLATFORMS = ["linux-64", "linux-aarch64", "osx-arm64", "win-64"]

# ---- Conda matrices ----

CONDA_BUILD_MATRIX = [
    {"python-version": "3.10", "numpy_build": "2.0"},
    {"python-version": "3.11", "numpy_build": "2.0"},
    {"python-version": "3.12", "numpy_build": "2.0"},
    {"python-version": "3.13", "numpy_build": "2.1"},
    {"python-version": "3.14", "numpy_build": "2.3"},
]

CONDA_TEST_MATRIX = [
    {"python-version": "3.10", "numpy_test": "1.23"},
    {"python-version": "3.10", "numpy_test": "1.24"},
    {"python-version": "3.10", "numpy_test": "1.25"},
    {"python-version": "3.11", "numpy_test": "1.26"},
    {"python-version": "3.11", "numpy_test": "2.0"},
    {"python-version": "3.11", "numpy_test": "2.2"},
    {"python-version": "3.12", "numpy_test": "1.26"},
    {"python-version": "3.12", "numpy_test": "2.0"},
    {"python-version": "3.12", "numpy_test": "2.2"},
    {"python-version": "3.13", "numpy_test": "2.2"},
    {"python-version": "3.13", "numpy_test": "2.3"},
    {"python-version": "3.14", "numpy_test": "2.4"},
]

# ---- Wheel matrices ----


def _python_tag(py_version):
    """Derive CPython wheel tag: '3.14t' -> 'cp314t'."""
    base = py_version.replace(".", "").replace("t", "")
    return "cp" + base + ("t" if py_version.endswith("t") else "")


def _canonical_version(py_version):
    """Strip free-threaded suffix: '3.14t' -> '3.14'."""
    return py_version.rstrip("t")


def _make_wheel_build_entry(py_version, numpy_build):
    return {
        "python-version": py_version,
        "numpy_build": numpy_build,
        "python_tag": _python_tag(py_version),
        "python_canonical": _canonical_version(py_version),
    }


def _make_wheel_test_entry(py_version, numpy_test):
    return {
        "python-version": py_version,
        "numpy_test": numpy_test,
        "python_canonical": _canonical_version(py_version),
    }


WHEEL_BUILD_MATRIX = [
    _make_wheel_build_entry("3.10", "2.0.2"),
    _make_wheel_build_entry("3.11", "2.0.2"),
    _make_wheel_build_entry("3.12", "2.0.2"),
    _make_wheel_build_entry("3.13", "2.1.3"),
    _make_wheel_build_entry("3.14", "2.3.3"),
    _make_wheel_build_entry("3.14t", "2.3.3"),
]

WHEEL_TEST_MATRIX = [
    _make_wheel_test_entry("3.10", "1.23"),
    _make_wheel_test_entry("3.10", "1.24"),
    _make_wheel_test_entry("3.10", "1.25"),
    _make_wheel_test_entry("3.11", "1.26"),
    _make_wheel_test_entry("3.11", "2.0"),
    _make_wheel_test_entry("3.11", "2.2"),
    _make_wheel_test_entry("3.12", "1.26"),
    _make_wheel_test_entry("3.12", "2.0"),
    _make_wheel_test_entry("3.12", "2.2"),
    _make_wheel_test_entry("3.13", "2.2"),
    _make_wheel_test_entry("3.13", "2.3"),
    _make_wheel_test_entry("3.14", "2.4"),
    _make_wheel_test_entry("3.14t", "2.4"),
]

# ---- Unified evaluate ----

_MATRICES = {
    "conda": (CONDA_BUILD_MATRIX, CONDA_TEST_MATRIX),
    "wheel": (WHEEL_BUILD_MATRIX, WHEEL_TEST_MATRIX),
}

_LABELS = {
    "conda": "build_numba_conda",
    "wheel": "build_numba_wheel",
}


def _add_platform(matrix, platform):
    return [dict(row, platform=platform) for row in matrix]


def evaluate(pkg_type, event, label, platform, inputs="{}"):
    """Return (build_matrix, test_matrix) for the given parameters.

    pkg_type: "conda" or "wheel"
    """
    base_build, base_test = _MATRICES[pkg_type]
    expected_label = _LABELS[pkg_type]

    if event in ("pull_request", "push"):
        build_matrix = list(base_build)
        test_matrix = list(base_test)
    elif event == "label" and label == expected_label:
        build_matrix = list(base_build)
        test_matrix = list(base_test)
    elif event == "workflow_dispatch":
        params = json.loads(inputs)
        build_matrix = list(base_build)
        test_matrix = list(base_test)

        python_version = params.get("python_version")
        if python_version:
            build_matrix = [
                r for r in build_matrix
                if r["python-version"] == python_version
            ]
            test_matrix = [
                r for r in test_matrix
                if r["python-version"] == python_version
            ]

        np_build_ver = params.get("numpy_build_version")
        if np_build_ver:
            build_matrix = [
                r for r in build_matrix
                if r["numpy_build"] == np_build_ver
            ]

        np_test_ver = params.get("numpy_test_version")
        if np_test_ver:
            test_matrix = [
                r for r in test_matrix
                if r["numpy_test"] == np_test_ver
            ]
    else:
        build_matrix = []
        test_matrix = []

    build_matrix = _add_platform(build_matrix, platform)
    test_matrix = _add_platform(test_matrix, platform)
    return build_matrix, test_matrix


if __name__ == "__main__":
    pkg_type = os.environ["GITHUB_PKG_TYPE"]
    event = os.environ.get("GITHUB_EVENT_NAME")
    label = os.environ.get("GITHUB_LABEL_NAME")
    inputs = os.environ.get("GITHUB_WORKFLOW_INPUT", "{}")
    platform = os.environ.get("GITHUB_PLATFORM")

    print(
        f"pkg_type='{pkg_type}', event='{event}', "
        f"label='{label}', platform='{platform}'"
    )

    build_matrix, test_matrix = evaluate(
        pkg_type, event, label, platform, inputs,
    )

    build_json = json.dumps(build_matrix)
    test_json = json.dumps(test_matrix)

    print(f"Build Matrix JSON: {build_json}")
    print(f"Test Matrix JSON: {test_json}")

    Path(os.environ["GITHUB_OUTPUT"]).write_text(
        f"build-matrix-json={build_json}\n"
        f"test-matrix-json={test_json}\n"
    )
