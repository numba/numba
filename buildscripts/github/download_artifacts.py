#!/usr/bin/env python
"""Download artifacts from GitHub Actions workflow runs for given run IDs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def get_artifacts_for_run(run_id, repo, token):
    """
    Get list of artifacts for a workflow run.

    Args:
        run_id: Workflow run ID
        repo: Repository in format 'owner/repo'
        token: GitHub token for API authentication

    Returns:
        List of artifact dictionaries with 'name' and 'id'
    """
    print(f"Fetching artifacts for run ID {run_id}", file=sys.stderr)

    env = os.environ.copy()
    env['GH_TOKEN'] = token

    try:
        result = subprocess.run(
            [
                "gh", "api",
                "-H", "Accept: application/vnd.github+json",
                "-H", "X-GitHub-Api-Version: 2022-11-28",
                f"/repos/{repo}/actions/runs/{run_id}/artifacts",
                "--jq", '.artifacts[] | {name: .name, id: .id}'
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        artifacts = []
        for line in result.stdout.strip().split('\n'):
            if line:
                artifacts.append(json.loads(line))

        print(f"Found {len(artifacts)} artifacts for run {run_id}", file=sys.stderr)
        return artifacts

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to fetch artifacts: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return []


def download_artifact(artifact_name, artifact_id, repo, token, output_dir):
    """
    Download a single artifact and extract directly to output_dir.

    Args:
        artifact_name: Name of the artifact
        artifact_id: Artifact ID
        repo: Repository in format 'owner/repo'
        token: GitHub token for API authentication
        output_dir: Directory to download artifact to

    Returns:
        True if successful, False otherwise
    """
    print(f"Downloading artifact: {artifact_name}", file=sys.stderr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['GH_TOKEN'] = token

    try:
        # Use gh CLI to download artifact
        zip_path = output_path / f"{artifact_name}.zip"
        result = subprocess.run(
            [
                "gh", "api",
                "-H", "Accept: application/vnd.github+json",
                "-H", "X-GitHub-Api-Version: 2022-11-28",
                f"/repos/{repo}/actions/artifacts/{artifact_id}/zip"
            ],
            check=True,
            env=env,
            capture_output=True
        )

        # Write the binary output to file
        with open(zip_path, 'wb') as f:
            f.write(result.stdout)

        # Unzip artifact directly to output_dir (flat structure)
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(output_path)],
            check=True
        )

        # Remove the zip file
        zip_path.unlink()

        print(f"Successfully downloaded: {artifact_name}", file=sys.stderr)
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to download {artifact_name}: {e}", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
        if e.stdout:
            print(f"stdout: {e.stdout}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download artifacts from workflow runs'
    )
    parser.add_argument(
        '--run-ids',
        nargs='+',
        required=True,
        help='List of workflow run IDs to download artifacts from'
    )
    parser.add_argument(
        '--repo',
        default=os.environ.get('GITHUB_REPOSITORY', ''),
        help='Repository in format owner/repo (default: GITHUB_REPOSITORY env var)'
    )
    parser.add_argument(
        '--token',
        default=os.environ.get('GH_TOKEN', os.environ.get('GITHUB_TOKEN', '')),
        help='GitHub token for API authentication (default: GH_TOKEN or GITHUB_TOKEN env var)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for artifacts'
    )

    args = parser.parse_args()

    if not args.repo:
        parser.error("Repository is required (via --repo or GITHUB_REPOSITORY env var)")
    if not args.token:
        parser.error("GitHub token is required (via --token or GH_TOKEN/GITHUB_TOKEN env var)")

    print(f"=== Downloading artifacts from {len(args.run_ids)} workflow runs ===", file=sys.stderr)

    for run_id in args.run_ids:
        print(f"\n--- Processing run ID: {run_id} ---", file=sys.stderr)
        artifacts = get_artifacts_for_run(run_id, args.repo, args.token)

        if not artifacts:
            print(f"\nERROR: No artifacts found for run ID {run_id}", file=sys.stderr)
            sys.exit(1)

        for artifact in artifacts:
            success = download_artifact(
                artifact['name'],
                artifact['id'],
                args.repo,
                args.token,
                args.output_dir
            )

            if not success:
                print(f"\nERROR: Failed to download artifact '{artifact['name']}'", file=sys.stderr)
                sys.exit(1)

    print(f"\n All artifacts downloaded successfully", file=sys.stderr)

if __name__ == "__main__":
    main()

