#!/usr/bin/env python
"""Declarative build/test matrices for GHA workflows.

Conda and wheel matrices are defined separately so either can change
independently.  Platform is added at evaluate time since all platforms
currently share the same version sets.

When run directly, reads GHA env vars and writes matrices to
$GITHUB_OUTPUT.
"""

import json
import os
from pathlib import Path

PLATFORMS = ["linux-64", "linux-aarch64", "osx-arm64", "win-64"]


def _canonical_version(py_version):
    """MAJOR.MINOR form, stripping free-threaded 't' and any patch level.

    '3.14' -> '3.14', '3.14t' -> '3.14', '3.14.3' -> '3.14',
    '3.14.3t' -> '3.14'.
    """
    base = py_version.rstrip("t")
    return ".".join(base.split(".")[:2])


def _python_tag(py_version):
    """CPython wheel tag from MAJOR.MINOR, preserving 't' suffix.

    '3.14' -> 'cp314', '3.14t' -> 'cp314t', '3.14.3' -> 'cp314',
    '3.14.3t' -> 'cp314t'.
    """
    base = _canonical_version(py_version).replace(".", "")
    return "cp" + base + ("t" if py_version.endswith("t") else "")


def _entry(py_version, numpy_key, numpy_value, with_tag=False):
    row = {
        "python-version": py_version,
        "python_canonical": _canonical_version(py_version),
        numpy_key: numpy_value,
    }
    if with_tag:
        row["python_tag"] = _python_tag(py_version)
    return row


# _entry() expands each (py, np) tuple into a matrix row.  Fully-resolved
# wheel-build row (evaluate() appends `platform`):
#   {"python-version": "3.14t", "python_canonical": "3.14",
#    "numpy_build": "2.3.3", "python_tag": "cp314t",
#    "platform": "linux-64"}
# Conda / wheel-test rows are the same minus `python_tag`.
#
# Field usage in .github/workflows/*.yml:
#   python-version   raw spec; setup-python input, conda-build --python=,
#                    wheel artifact names, `endsWith(.., 't')` gates.
#   python_canonical MAJOR.MINOR (no 't'); llvmlite artifact name (shared
#                    by ft / non-ft), conda numba artifact name, and
#                    `python<MAJOR.MINOR>` binary lookup on Linux.
#   python_tag       cpXY[t]; only manylinux builders, to form the
#                    /opt/python/<ver_tag>-<python_tag>/bin/python path.
#   numpy_build      pinned numpy in build env (pip / conda-build --numpy=).
#   numpy_test       pinned numpy in test env.
#   platform         tagged on every row but unused in YAML today (one
#                    workflow per platform); kept for downstream consumers.

# ---- Conda matrices ----

CONDA_BUILD_MATRIX = [
    _entry(py, "numpy_build", np) for py, np in (
        ("3.10", "2.0"),
        ("3.11", "2.0"),
        ("3.12", "2.0"),
        ("3.13", "2.1"),
        ("3.14", "2.3"),
    )
]

CONDA_TEST_MATRIX = [
    _entry(py, "numpy_test", np) for py, np in (
        ("3.10", "1.23"),
        ("3.10", "1.24"),
        ("3.10", "1.25"),
        ("3.11", "1.26"),
        ("3.11", "2.0"),
        ("3.11", "2.2"),
        ("3.12", "1.26"),
        ("3.12", "2.0"),
        ("3.12", "2.2"),
        ("3.13", "2.2"),
        ("3.13", "2.3"),
        ("3.14", "2.4"),
    )
]

# ---- Wheel matrices ----

WHEEL_BUILD_MATRIX = [
    _entry(py, "numpy_build", np, with_tag=True) for py, np in (
        ("3.10", "2.0.2"),
        ("3.11", "2.0.2"),
        ("3.12", "2.0.2"),
        ("3.13", "2.1.3"),
        ("3.14", "2.3.3"),
        ("3.14t", "2.3.3"),
    )
]

WHEEL_TEST_MATRIX = [
    _entry(py, "numpy_test", np) for py, np in (
        ("3.10", "1.23"),
        ("3.10", "1.24"),
        ("3.10", "1.25"),
        ("3.11", "1.26"),
        ("3.11", "2.0"),
        ("3.11", "2.2"),
        ("3.12", "1.26"),
        ("3.12", "2.0"),
        ("3.12", "2.2"),
        ("3.13", "2.2"),
        ("3.13", "2.3"),
        ("3.14", "2.4"),
        ("3.14t", "2.4"),
    )
]

# ---- Unified evaluate ----

_MATRICES = {
    "conda": (CONDA_BUILD_MATRIX, CONDA_TEST_MATRIX),
    "wheel": (WHEEL_BUILD_MATRIX, WHEEL_TEST_MATRIX),
}

# Build-trigger labels on a PR.  When any of these are applied, the script
# acts as a selector: only the matching pkg_type's matrix is emitted, others
# return empty.  When no build_numba_* label is present we fall back to the
# historical default of emitting the full matrix for the triggering event.
_LABELS = {
    "conda": "build_numba_conda",
    "wheel": "build_numba_wheel",
}
_BUILD_LABELS = frozenset(_LABELS.values())


def _filter(rows, key, value):
    return [r for r in rows if r.get(key) == value] if value else rows


def evaluate(pkg_type, event, pr_labels, platform, inputs="{}"):
    """Return (build_matrix, test_matrix) for the given parameters.

    pkg_type:  "conda" or "wheel"
    event:     `github.event_name` (e.g. "pull_request", "push",
               "schedule", "workflow_dispatch")
    pr_labels: list of label names currently on the PR.  Empty list for
               non-PR events.  When a `pull_request: types: [labeled]`
               trigger fires, GHA still reports event_name=pull_request;
               the per-event label is in pr_labels along with all others.
    """
    base_build, base_test = _MATRICES[pkg_type]
    my_label = _LABELS[pkg_type]
    selected = set(pr_labels or ()) & _BUILD_LABELS

    if event == "pull_request":
        accept_full = my_label in selected if selected else True
    elif event == "schedule":
        accept_full = True
    else:
        accept_full = False

    if accept_full:
        build, test = list(base_build), list(base_test)
    elif event == "workflow_dispatch":
        params = json.loads(inputs)
        pv = params.get("python_version")
        build = _filter(list(base_build), "python-version", pv)
        test = _filter(list(base_test), "python-version", pv)
        build = _filter(build, "numpy_build",
                        params.get("numpy_build_version"))
        test = _filter(test, "numpy_test",
                       params.get("numpy_test_version"))
    else:
        build, test = [], []

    build = [dict(r, platform=platform) for r in build]
    test = [dict(r, platform=platform) for r in test]
    return build, test


if __name__ == "__main__":
    pkg_type = os.environ["GITHUB_PKG_TYPE"]
    event = os.environ.get("GITHUB_EVENT_NAME")
    pr_labels = json.loads(os.environ.get("GITHUB_PR_LABELS") or "[]")
    inputs = os.environ.get("GITHUB_WORKFLOW_INPUT", "{}")
    platform = os.environ.get("GITHUB_PLATFORM")

    print(
        f"pkg_type='{pkg_type}', event='{event}', platform='{platform}', "
        f"pr_labels={pr_labels}"
    )

    build_matrix, test_matrix = evaluate(
        pkg_type, event, pr_labels, platform, inputs,
    )

    build_json = json.dumps(build_matrix)
    test_json = json.dumps(test_matrix)

    print(f"Build Matrix JSON: {build_json}")
    print(f"Test Matrix JSON: {test_json}")

    Path(os.environ["GITHUB_OUTPUT"]).write_text(
        f"build-matrix-json={build_json}\n"
        f"test-matrix-json={test_json}\n"
    )
