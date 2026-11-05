"""Render the sticky PR benchmark comment from merged result JSON."""

import argparse
import json
import sys
from pathlib import Path

MARKER = "<!-- numba-pr-benchmark -->"


def load_documents(directory):
    documents = {}
    for path in sorted(Path(directory).rglob("*.json")):
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "commit" in document and "measurements" in document:
            documents[document["commit"]] = document
    return documents


def env_parts(key):
    os_name, python, numpy = key.split("|", 2)
    return os_name, python, numpy


def env_label(key):
    os_name, python, numpy = env_parts(key)
    return f"{os_name} · Python {python} · NumPy {numpy}"


def short_label(key):
    os_name, python, numpy = env_parts(key)
    return f"{os_name} / py{python} / np{numpy}"


def milliseconds(ns):
    return f"{ns / 1e6:,.3f}"


def ratio_text(case):
    if not case:
        return "n/a"
    return f"{case['numba_ns'] / case['numpy_ns']:.2f}x"


def change_percent(case, baseline):
    if not baseline or not baseline.get("numba_ns"):
        return None
    return (case["numba_ns"] - baseline["numba_ns"]) / baseline["numba_ns"] * 100


def marked_change(delta):
    if delta is None:
        return "new"
    mark = "🟢" if delta < 0 else "🔴" if delta > 0 else "⚪"
    return f"{mark} {delta:+.1f}%"


def case_order(case):
    return (case.get("order", sys.maxsize), case["id"])


def time_text(case):
    return milliseconds(case["numba_ns"]) if case else "n/a"


def notable_table(rows):
    lines = [
        "| Case | System | Numba main ms | Numba PR ms | NumPy ms | "
        "Numba / NumPy main | Numba / NumPy PR | Numba change |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for delta, key, case, baseline in rows:
        lines.append(
            f"| `{case['id']}` | {short_label(key)} | "
            f"{time_text(baseline)} | {milliseconds(case['numba_ns'])} | "
            f"{milliseconds(case['numpy_ns'])} | {ratio_text(baseline)} | "
            f"{ratio_text(case)} | {marked_change(delta)} |"
        )
    return lines


def system_table(entry, baselines):
    lines = [
        "| Case | Numba main ms | Numba PR ms | NumPy ms | "
        "Numba / NumPy main | Numba / NumPy PR | Numba change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sorted(entry["cases"], key=case_order):
        baseline = baselines.get(case["id"])
        delta = change_percent(case, baseline)
        lines.append(
            f"| `{case['id']}` | {time_text(baseline)} | "
            f"{milliseconds(case['numba_ns'])} | {milliseconds(case['numpy_ns'])} | "
            f"{ratio_text(baseline)} | {ratio_text(case)} | "
            f"{marked_change(delta)} |"
        )
    return lines


def skip_text(skip):
    reason = skip.get("reason") or f"requires numba {skip.get('requires_numba')}"
    return f"`{skip['id']}` ({reason})"


def render(args):
    documents = load_documents(args.results)
    head = documents.get(args.head)
    base = documents.get(args.base) if args.base else None
    if head is not None and base is None:
        candidates = [
            (len(document["measurements"]), commit, document)
            for commit, document in documents.items()
            if commit != args.head
        ]
        if candidates:
            base = max(candidates, key=lambda item: item[0])[2]
    expected = [key for key in args.expected.split(",") if key]
    title = "## Numba PR benchmarks"
    if args.pr:
        title += f" (#{args.pr})"
    lines = [MARKER, title, ""]

    if head is None:
        lines.append(
            f"No benchmark results for `{args.head[:10]}`. "
            f"See the [workflow run]({args.run_url}) for details."
        )
        return "\n".join(lines) + "\n"

    head_commit = head["commit"][:10]
    head_repo = args.head_repo or "numba/numba"
    base_repo = args.base_repo or "numba/numba"
    head_link = f"https://github.com/{head_repo}/commit/{args.head}"
    if base is not None:
        base_commit = base["commit"][:10]
        base_link = f"https://github.com/{base_repo}/commit/{base['commit']}"
        lines.append(
            f"**Head:** [`{head_repo}@{head_commit}`]({head_link}) · "
            f"**Baseline:** [`{base_repo}@{base_commit}`]({base_link})"
        )
    else:
        lines.append(f"**Head:** [`{head_repo}@{head_commit}`]({head_link})")

    measured = list(head["measurements"])
    ordered = expected + [key for key in measured if key not in expected]
    present = [key for key in ordered if key in head["measurements"]]
    missing = [key for key in ordered if key not in head["measurements"]]

    base_cases = {}
    if base is not None:
        for key, entry in base["measurements"].items():
            base_cases[key] = {case["id"]: case for case in entry["cases"]}

    notable = []
    for key in present:
        for case in head["measurements"][key]["cases"]:
            baseline = base_cases.get(key, {}).get(case["id"])
            delta = change_percent(case, baseline)
            if delta is not None and abs(delta) >= args.threshold:
                notable.append((delta, key, case, baseline))
    notable.sort(key=lambda item: item[0], reverse=True)

    regressions = sum(1 for item in notable if item[0] > 0)
    improvements = sum(1 for item in notable if item[0] < 0)
    summary = (
        f"**Systems:** {len(present)} measured · "
        f"🔴 {regressions} regression(s) · 🟢 {improvements} improvement(s) "
        f"(≥{args.threshold:g}%)"
    )
    if args.run_url:
        summary += f" · [workflow run]({args.run_url})"
    lines.append(summary)
    lines.append(
        "_Numba / NumPy is Numba time divided by NumPy time (lower is better). "
        "Numba change is PR time vs main; 🟢 faster, 🔴 slower._"
    )

    if missing:
        lines.append(
            "**Not measured:** " + ", ".join(env_label(key) for key in missing)
        )
    if base is None:
        lines.append("**Baseline results unavailable; changes cannot be computed.**")
    lines.append("")

    lines.append(f"### Regressions and improvements (≥{args.threshold:g}%)")
    lines.append("")
    if notable:
        lines.extend(notable_table(notable))
    else:
        lines.append(f"No benchmark moved by at least {args.threshold:g}%.")
    lines.append("")

    for key in present:
        entry = head["measurements"][key]
        lines.append("<details>")
        lines.append(
            f"<summary><b>{env_label(key)}</b> — "
            f"{len(entry['cases'])} benchmark(s)</summary>"
        )
        lines.append("")
        lines.extend(system_table(entry, base_cases.get(key, {})))
        skipped = sorted(entry.get("skipped") or [], key=case_order)
        if skipped:
            lines.append("")
            lines.append("Skipped: " + ", ".join(skip_text(skip) for skip in skipped))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--pr", default="")
    parser.add_argument("--head-repo", default="numba/numba")
    parser.add_argument("--base-repo", default="numba/numba")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--expected", default="")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    text = render(args)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
