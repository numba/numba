#!/usr/bin/env python3
"""Generate a release Pull-Requests / Authors list for Numba.

Auto-detects the latest ``X.Y.0dev0`` tag as the start point, lists the
merged PRs since then (skipping any already in ``docs/source/release``
or ``CHANGE_LOG``), and credits every author including
``Co-authored-by:`` trailers.

Prints to stdout by default; pass ``--write`` to append the list to
``--changelog`` without changing any existing header or whitespace.
Token comes from ``--token``, ``$GITHUB_TOKEN``/``$GH_TOKEN``, or
``gh auth token``.

Examples:
  python maint/generate_changelog.py
  python maint/generate_changelog.py --start 0.68.0dev0
  python maint/generate_changelog.py --start 0.68.0dev0 --write
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

from github import Github, Auth, GithubException

CHANGE_LOG = "CHANGE_LOG"
RELEASE_NOTES_DIR = Path("docs/source/release")
COAUTHOR_RE = re.compile(r"^Co-authored-by:\s*(.+?)\s*<(.+?)>", re.MULTILINE)
_EMAIL_CACHE = {}
# GitHub account logins, and the git committer name "GitHub", should
# not be credited. web-flow is the account behind web UI merges.
_SKIP_LOGINS = {"web-flow", "GitHub"}
_SKIP_EMAILS = {"noreply@github.com"}


def sh(*cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False
    ).stdout.strip()


def detect_start():
    """Latest X.Y.0dev0 tag (Numba) or vX.Y.0dev0 (llvmlite)."""
    tags = sh("git", "tag", "-l", "*dev0", "--sort=-v:refname").split("\n")
    return next((t for t in tags if t), None)


def merged_pr_numbers(start):
    log = sh("git", "log", f"{start}..HEAD", "--oneline",
             "--grep", "Merge pull request")
    tagged = sh("git", "log", "-1", "--oneline", "--grep",
                "Merge pull request", start)
    lines = [ln for ln in f"{log}\n{tagged}".split("\n") if ln]
    nums = {int(m.group(1)) for line in lines
            if (m := re.search(r"#(\d+)", line))}
    return sorted(nums)


def default_changelog():
    notes = sorted(RELEASE_NOTES_DIR.glob("*.0-notes.rst"))
    return str(notes[-1]) if notes else CHANGE_LOG


def _prs_in_text(text):
    return {int(n) for n in re.findall(r"PR `#(\d+)", text)}


def known_pr_numbers(path):
    known = set()
    paths = [Path(path)]
    if RELEASE_NOTES_DIR.is_dir():
        paths.extend(RELEASE_NOTES_DIR.glob("*.rst"))
    for p in paths:
        try:
            known |= _prs_in_text(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
    return known


def resolve_token(token):
    token = token or os.environ.get("GITHUB_TOKEN") or \
        os.environ.get("GH_TOKEN") or sh("gh", "auth", "token")
    assert token, ("No GitHub token: pass --token, set GITHUB_TOKEN, or "
                   "run `gh auth login`.")
    return token


def _unresolvable_email(email):
    """True only for addresses in ``_SKIP_EMAILS``."""
    return email.lower() in _SKIP_EMAILS


def resolve_email(gh, email):
    """Map a co-author email to a GitHub user (login, url); name-only if not."""
    if email in _EMAIL_CACHE:
        return _EMAIL_CACHE[email]
    user = None
    try:
        if email.endswith("@users.noreply.github.com"):
            login = email.split("@")[0].split("+")[-1]
            user = gh.get_user(login)
        elif not _unresolvable_email(email):
            hits = gh.search_users(f"{email} in:email")
            if hits.totalCount:
                candidate = hits[0]
                # in:email search is fuzzy and may return unrelated
                # users; only accept an exact public-email match.
                if (candidate.email or "").lower() == email.lower():
                    user = candidate
    except GithubException:
        user = None
    _EMAIL_CACHE[email] = user
    return user


def gh_identity(user):
    if user is None:
        return None
    try:
        login = user.login
        if login in _SKIP_LOGINS:
            return None
        return (login, user.html_url)
    except GithubException:
        return None


def git_identity(gh, git_user):
    if git_user is None:
        return None
    if git_user.email:
        user = resolve_email(gh, git_user.email)
        if user is not None:
            ident = gh_identity(user)
            if ident:
                return ident
    if git_user.name and git_user.name not in _SKIP_LOGINS:
        return (git_user.name, None)
    return None


def person_identity(gh, gh_user, git_user):
    """Identity for a commit author or committer.

    A login in ``_SKIP_LOGINS`` (e.g. web-flow) means the git identity
    carries no credit information, so the git fallback is skipped too.
    """
    if getattr(gh_user, "login", None) in _SKIP_LOGINS:
        return None
    return gh_identity(gh_user) or git_identity(gh, git_user)


def pr_authors(gh, pr):
    """Set of (login_or_name, url_or_None) including co-author trailers."""
    authors = set()
    for c in pr.get_commits():
        for gh_user, git_user in ((c.author, c.commit.author),
                                  (c.committer, c.commit.committer)):
            ident = person_identity(gh, gh_user, git_user)
            if ident:
                authors.add(ident)
        for name, email in COAUTHOR_RE.findall(c.commit.message):
            user = resolve_email(gh, email)
            ident = gh_identity(user) if user else None
            if not ident and name not in _SKIP_LOGINS:
                ident = (name, None)
            if ident:
                authors.add(ident)
    return authors


def credit(login, url):
    return f"`{login} <{url}>`_" if url else login


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", help="Start tag/commit "
                        "(default: latest *dev0 tag)")
    parser.add_argument("--token", help="GitHub token (default: env/gh)")
    parser.add_argument("--repo", default="numba/numba")
    parser.add_argument("--changelog", default=None,
                        help="Target notes file (default: latest "
                             "docs/source/release/*.0-notes.rst)")
    parser.add_argument("--write", action="store_true",
                        help="Append the PR/author list to --changelog")
    args = parser.parse_args()
    if args.changelog is None:
        args.changelog = default_changelog()

    start = args.start or detect_start()
    assert start, "Could not detect a *dev0 tag; pass --start."
    print(f"Start point: {start}", file=sys.stderr)

    known = known_pr_numbers(args.changelog)
    merged = merged_pr_numbers(start)
    fresh = [n for n in merged if n not in known]
    skipped = [n for n in merged if n in known]
    if skipped:
        print(
            f"Skipping {len(skipped)} already in changelog: "
            f"{', '.join('#%d' % n for n in skipped)}",
            file=sys.stderr
        )
    assert fresh, "No new PRs to add after dedup."
    print(f"Including {len(fresh)} new PR(s)", file=sys.stderr)

    gh = Github(auth=Auth.Token(resolve_token(args.token)))
    repo = gh.get_repo(args.repo)

    pr_lines = []
    all_authors = set()
    for i, num in enumerate(fresh, 1):
        print(f"  [{i}/{len(fresh)}] PR #{num}", file=sys.stderr)
        pr = repo.get_pull(num)
        authors = pr_authors(gh, pr)
        all_authors |= authors
        names = " ".join(credit(*a) for a in
                         sorted(authors, key=lambda a: a[0].lower()))
        pr_lines.append(f"* PR `#{num} <{pr.html_url}>`_: {pr.title} ({names})")

    author_lines = [f"* {credit(*a)}" for a in
                    sorted(all_authors, key=lambda a: a[0].lower())]

    body = ("Pull-Requests:\n\n" + "\n".join(pr_lines) +
            "\n\nAuthors:\n\n" + "\n".join(author_lines) + "\n")

    if args.write:
        assert os.path.exists(args.changelog), (
            f"{args.changelog} not found; run towncrier first.")
        with open(args.changelog, encoding="utf-8") as fh:
            existing = fh.read()
        sep = "" if existing.endswith("\n") else "\n"
        with open(args.changelog, "w", encoding="utf-8") as fh:
            fh.write(existing + sep + "\n" + body)
        print(f"Appended PR/author list to {args.changelog}", file=sys.stderr)
    else:
        print("\n" + body)


if __name__ == "__main__":
    main()
