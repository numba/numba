#!/usr/bin/env python
"""Find successful llvmlite workflow run IDs for a given git tag."""

import argparse
import json
import os
import subprocess
import sys


def find_workflow_run(workflow_name, tag, repo, token):
    """
    Find the latest successful workflow run for a given workflow and tag.

    Args:
        workflow_name: Name of the workflow file (e.g., 'llvmlite_conda_builder.yml')
        tag: Git tag to search for
        repo: Repository in format 'owner/repo' (e.g., 'numba/llvmlite')
        token: GitHub token for API authentication

    Returns:
        Run ID as a string, or None if not found
    """
    print(f"Search for successful run of {workflow_name} on tag {tag} in {repo}", file=sys.stderr)

    env = os.environ.copy()
    env['GH_TOKEN'] = token

    try:
        result = subprocess.run(
            [
                "gh", "api",
                "-H", "Accept: application/vnd.github+json",
                "-H", "X-GitHub-Api-Version: 2022-11-28",
                f"/repos/{repo}/actions/workflows/{workflow_name}/runs",
                "--jq", f'.workflow_runs[] | select(.head_branch == "{tag}" and .conclusion == "success") | .id'
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        run_ids = result.stdout.strip().split('\n')
        if run_ids and run_ids[0]:
            run_id = run_ids[0]
            print(f"Found run ID {run_id} for {workflow_name}", file=sys.stderr)
            return run_id

        print(f"ERROR: No successful run found for {workflow_name} on tag {tag}", file=sys.stderr)
        return None

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to query GitHub API: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None


def load_workflow_config(config_path):
    """
    Load workflow groups from JSON config.

    Args:
        config_path: Path to JSON file containing workflow groups under 'workflows' key

    Returns:
        Dict mapping group names to lists of workflow filenames
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        workflow_groups = config['workflows']
    except KeyError as e:
        print(f"ERROR: Missing required key in config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate that all groups are lists
    for group_name, workflows in workflow_groups.items():
        if not isinstance(workflows, list):
            print(f"ERROR: Workflow group '{group_name}' must be a list", file=sys.stderr)
            sys.exit(1)

    return workflow_groups


def main():
    parser = argparse.ArgumentParser(
        description='Find successful workflow run IDs for a given tag'
    )
    parser.add_argument(
        '--tag',
        default=os.environ.get('TAG', ''),
        help='Git tag to search for (default: TAG env var)'
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
        '--config',
        required=True,
        help='Path to JSON config file containing workflow groups'
    )

    args = parser.parse_args()

    if not args.tag:
        parser.error("TAG is required (via --tag or TAG env var)")
    if not args.repo:
        parser.error("Repository is required (via --repo or GITHUB_REPOSITORY env var)")
    if not args.token:
        parser.error("GitHub token is required (via --token or GH_TOKEN/GITHUB_TOKEN env var)")

    # Load workflow groups from config file
    # Returns dict: {group_name: [workflow_file1.yml, workflow_file2.yml, ...]}
    grouped_workflow_names = load_workflow_config(args.config)

    # Will store found run IDs: {group_name: [run_id1, run_id2, ...]}
    found_run_ids_by_group = {}

    # For each workflow group, query GitHub API to find successful run IDs
    # matching the specified tag
    for group_name, workflow_files in grouped_workflow_names.items():
        print(f"\n=== finding {group_name} workflow runs ===", file=sys.stderr)
        run_ids = []
        for workflow_file in workflow_files:
            run_id = find_workflow_run(workflow_file, args.tag, args.repo, args.token)
            if run_id:
                run_ids.append(run_id)
            else:
                print(f"WARNING: Could not find run ID for {workflow_file}", file=sys.stderr)
        found_run_ids_by_group[group_name] = run_ids

    # Verify all workflows were found
    total_expected = sum(len(workflows) for workflows in grouped_workflow_names.values())
    total_found = sum(len(run_ids) for run_ids in found_run_ids_by_group.values())

    if total_found != total_expected:
        print(f"\nERROR: Not all workflow runs were found ({total_found}/{total_expected})", file=sys.stderr)
        sys.exit(1)

    print("\n=== found workflow run IDs ===", file=sys.stderr)
    for group_name, run_ids in found_run_ids_by_group.items():
        print(f"{group_name}: {' '.join(run_ids)}", file=sys.stderr)

    # Output run IDs in shell variable format for GitHub Actions
    # eg. wheel_run_ids=run_id1 run_id2 ...
    for group_name, run_ids in found_run_ids_by_group.items():
        print(f"{group_name}_run_ids={' '.join(run_ids)}")


if __name__ == "__main__":
    main()

