#!/usr/bin/env python3
"""
Fetch Azure Pipelines build logs for a numba/numba pull request.

Numba's CI runs on Azure Pipelines at https://dev.azure.com/numba/numba.
When a PR is opened on GitHub, Azure builds the synthetic branch
``refs/pull/<PR_NUMBER>/merge``. This script queries the Azure DevOps
REST API to find those builds and download their per-job logs.

The project is public, so no credentials are required. If you hit
anonymous rate limits you can set an AZURE_DEVOPS_PAT env var with a
Personal Access Token (Build: Read scope is sufficient).

Examples:
    # Download logs for the most recent build of PR 9876
    python azure_logs.py 9876

    # Download logs for the most recent build of PR 9876 into a custom directory
    python azure_logs.py 9876 --out-dir my_logs/pr
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator

import requests

ORG = "numba"
PROJECT = "numba"
API = f"https://dev.azure.com/{ORG}/{PROJECT}/_apis"
API_VERSION = "7.1"

# Matches numba's per-test memory profiling lines, e.g.:
#   2026-04-22T19:31:46.7334303Z Name: numba.tests.test_x.Foo.test_y | PID: 10104
#   | Start: 19:30:29 | Duration: 0.119s | Start RSS: 245.88 MB
#   | End RSS: 246.93 MB | RSS delta: +1.04 MB | Avail memory: 6.39 GB
TEST_MEM_RE = re.compile(
    r"^(?P<timestamp>\S+)\s+"
    r"Name:\s+(?P<name>\S+)\s+\|\s+"
    r"PID:\s+(?P<pid>\d+)\s+\|\s+"
    r"Start:\s+(?P<start>\d{2}:\d{2}:\d{2})\s+\|\s+"
    r"Duration:\s+(?P<duration>[\d.]+)s\s+\|\s+"
    r"Start RSS:\s+(?P<start_rss>[\d.]+)\s+MB\s+\|\s+"
    r"End RSS:\s+(?P<end_rss>[\d.]+)\s+MB\s+\|\s+"
    r"RSS delta:\s+(?P<rss_delta>[+-]?[\d.]+)\s+MB\s+\|\s+"
    r"Avail memory:\s+(?P<avail>[\d.]+)\s+(?P<avail_unit>[KMGT]?B)\s*$"
)

# Multipliers for converting the "Avail memory" field into MB.
_UNIT_TO_MB = {"B": 1 / (1024 * 1024), "KB": 1 / 1024, "MB": 1.0,
               "GB": 1024.0, "TB": 1024.0 * 1024.0}



def get_builds_for_pr(pr_number: int) -> list[dict]:
    """Return all Azure builds associated with a GitHub PR, newest first."""
    url = f"{API}/build/builds"
    params = {
        "branchName": f"refs/pull/{pr_number}/merge",
        "queryOrder": "queueTimeDescending",
        "api-version": API_VERSION,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])


def get_timeline(build_id: int) -> list[dict]:
    """Return the timeline records (jobs, phases, tasks) for a build."""
    url = f"{API}/build/builds/{build_id}/timeline"
    params = {"api-version": API_VERSION}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("records", [])


def get_log_text(build_id: int, log_id: int) -> str:
    """Return the raw text of a log stream."""
    url = f"{API}/build/builds/{build_id}/logs/{log_id}"
    params = {"api-version": API_VERSION}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.text


def parse_test_mem_lines(text: str, OS: str, label: str) -> Iterator[dict]:
    """Yield a typed dict for every per-test memory line in ``text``."""
    for line in text.splitlines():
        m = TEST_MEM_RE.match(line)
        if not m:
            continue
        avail_val = float(m["avail"])
        avail_unit = m["avail_unit"].upper()
        test_name = m["name"].split(".")[-1]

        status = "PASS"
        reason = ""
        if f"FAIL: {test_name}" in text:
            status = "FAIL"
            # Get the line number that contains FAIL: test_name
            fail_line_number = next((i for i, l in enumerate(text.splitlines()) if f"FAIL: {test_name}" in l), -1)
            if fail_line_number != -1:
                # Save the lines until next FAIL: or FAILED (failures=
                fail_lines = []
                for l in text.splitlines()[fail_line_number + 1:]:
                    if (f"FAIL: " in l) or ("FAILED (failures=" in l) or ("FAILED (errors=" in l) or ("ERROR: " in l):
                        break
                    fail_lines.append(l)
                # Add the fail lines to the reason, if any
                reason += "\n" + "\n".join(fail_lines)
        
        if f"ERROR: {test_name}" in text:
            status = "ERROR"
            # Get the line number that contains ERROR: test_name
            error_line_number = next((i for i, l in enumerate(text.splitlines()) if f"ERROR: {test_name}" in l), -1)
            if error_line_number != -1:
                # Save the lines until next ERROR: or FAILED (errors=
                error_lines = []
                for l in text.splitlines()[error_line_number + 1:]:
                    if (f"ERROR: " in l) or ("FAILED (errors=" in l) or ("FAILED (failures=" in l) or ("FAIL: " in l):
                        break
                    error_lines.append(l)
                # Add the error lines to the reason, if any
                reason += "\n" + "\n".join(error_lines)
    
        yield {
            "timestamp": m["timestamp"],
            "name": m["name"],
            "test_name": test_name,
            "pid": int(m["pid"]),
            "start": m["start"],
            "duration_s": float(m["duration"]),
            "start_rss_mb": float(m["start_rss"]),
            "end_rss_mb": float(m["end_rss"]),
            "rss_delta_mb": float(m["rss_delta"]),
            "avail_memory_mb": round(avail_val * _UNIT_TO_MB[avail_unit], 4),
            "avail_memory_raw": f"{m['avail']} {avail_unit}",
            "OS": OS,
            "label": label,
            "status": status,
            "reason": reason.strip(),
        }


def summarize(entries: list[dict], top_n: int = 100) -> dict:
    """Compute top-N rankings and totals over parsed test-mem entries."""
    if not entries:
        return {"count": 0}

    def top(key, reverse=True):
        ranked = sorted(entries, key=lambda e: e[key], reverse=reverse)[:top_n]
        return [{"name": e["name"], key: e[key]} for e in ranked]

    total_duration = sum(e["duration_s"] for e in entries)
    return {
        "count": len(entries),
        "total_duration_s": round(total_duration, 3),
        "mean_duration_s": round(total_duration / len(entries), 4),
        "max_end_rss_mb": max(e["end_rss_mb"] for e in entries),
        "slowest": top("duration_s"),
        "largest_rss_growth": top("rss_delta_mb"),
        "largest_rss_shrink": top("rss_delta_mb", reverse=False),
        "highest_end_rss": top("end_rss_mb"),
        "failed_tests": [e for e in entries if e["status"] == "FAIL" or e["status"] == "ERROR"],
    }


def print_summary(summary: dict, label: str = "") -> None:
    """Print a compact human-readable summary to stdout."""
    if summary.get("count", 0) == 0:
        print(f"  {label}(no matching test-mem lines)" if label
              else "  (no matching test-mem lines)")
        return
    header = f"Summary{f' ({label})' if label else ''}:"
    print(header)
    print(f"  tests recorded    : {summary['count']}")
    print(f"  total duration    : {summary['total_duration_s']}s")
    print(f"  mean duration     : {summary['mean_duration_s']}s")
    print(f"  peak end RSS      : {summary['max_end_rss_mb']} MB")
    print("  slowest tests:")
    for e in summary["slowest"][:50]:
        print(f"    {e['duration_s']:>8.3f}s  {e['name']}")
    print("  largest RSS growth:")
    for e in summary["largest_rss_growth"][:50]:
        print(f"    {e['rss_delta_mb']:>+8.2f} MB  {e['name']}")
    print("  failed tests:")
    for e in summary["failed_tests"]:
        print(f"    {e['name']}")
    print()



def sanitize(name: str) -> str:
    """Make a string safe for use as a filename component."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


def print_builds(pr_number: int, builds: list[dict]) -> None:
    print(f"Builds for numba/numba PR #{pr_number}:")
    if not builds:
        print("  (none found — PR may predate Azure CI, may not have triggered")
        print("   a build, or may be from a fork still awaiting CI approval.)")
        return
    print(f"  {'BuildID':>8}  {'Status':<12}  {'Result':<12}  "
          f"{'Queued (UTC)':<20}  Definition")
    print("  " + "-" * 86)
    for b in builds:
        print(
            f"  {b['id']:>8}  "
            f"{(b.get('status') or '-'):<12}  "
            f"{(b.get('result') or '-'):<12}  "
            f"{(b.get('queueTime') or '')[:19]:<20}  "
            f"{b.get('definition', {}).get('name', '-')}"
        )


def download_logs(build: dict, out_root: Path) -> None:
    """Download per-job logs for a build into out_root/pr-build-<id>/."""
    build_id = build["id"]
    out_dir = out_root / f"pr-build-{build_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop a small metadata file for context.
    (out_dir / "build_info.txt").write_text(
        f"Build        : {build_id}\n"
        f"Web URL      : {build.get('_links', {}).get('web', {}).get('href', '')}\n"
        f"Definition   : {build.get('definition', {}).get('name', '')}\n"
        f"Source branch: {build.get('sourceBranch', '')}\n"
        f"Source commit: {build.get('sourceVersion', '')}\n"
        f"Status       : {build.get('status', '')}\n"
        f"Result       : {build.get('result', '')}\n"
        f"Queued       : {build.get('queueTime', '')}\n"
        f"Started      : {build.get('startTime', '')}\n"
        f"Finished     : {build.get('finishTime', '')}\n"
    )

    records = get_timeline(build_id)
    # Job-level records carry the consolidated log most users want.
    jobs = [r for r in records if r.get("type") == "Job"]
    if not jobs:
        kind = ""
        print(f"  no {kind}jobs with logs found for build {build_id}")
        return

    print(f"  downloading {len(jobs)} job log(s) -> {out_dir}")
    all_entries: list[dict] = []
    for job in jobs:
        log = job.get("log") or {}
        log_id = log.get("id")
        if log_id is None:
            continue
        job_name = sanitize(job.get("name") or f"job-{job.get('id', '')}")
        result = job.get("result") or "unknown"
        out_path = out_dir / f"{job_name}__{result}.log"
        try:
            text = get_log_text(build_id, log_id)
        except requests.HTTPError as e:
            print(f"    ! {job_name}: {e}")
            continue
        out_path.write_text(text, encoding="utf-8", errors="replace")
        entries = list(parse_test_mem_lines(text, job_name.split(" ")[0], job_name.split(" ")[1]))

        all_entries.extend(entries)
        print(f"    - {out_path.name} ({len(text):,} bytes, "
              f"{len(entries)} test-mem line(s))")
    
    # Save all parsed entries across all jobs as JSON for further analysis.
    (out_dir / "test_logs.json").write_text(
        json.dumps(all_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Build-level roll-up across every job in this build.
    summary = summarize(all_entries)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_summary(summary, label=f"build {build_id}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Azure Pipelines build logs for a numba/numba PR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pr_number", type=int, nargs="?",
                   help="numba/numba PR number (omit when using --parse).")
    p.add_argument("--out-dir", default="numba_pr_logs",
                   help="Output directory (default: numba_pr_logs).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.pr_number is None:
        print("error: pr_number is required unless --parse is used.",
              file=sys.stderr)
        return 2

    try:
        builds = get_builds_for_pr(args.pr_number)
    except requests.HTTPError as e:
        print(f"Failed to query builds: {e}", file=sys.stderr)
        return 1

    if not builds:
        print(f"No builds found for PR {args.pr_number}; nothing to download.")
        return 0

    targets = builds[:1]
    out_root = Path(args.out_dir) / f"pr-{args.pr_number}"
    out_root.mkdir(parents=True, exist_ok=True)

    for b in targets:
        print(f"Build {b['id']}  result={b.get('result', 'n/a')}  "
              f"status={b.get('status', 'n/a')}")
        download_logs(b, out_root)

    print(f"\nDone. Logs under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())