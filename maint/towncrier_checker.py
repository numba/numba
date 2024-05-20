"""towncrier_checker.py

Find missing towncrier news fragments.

Usage:
  towncrier_checker.py (-h | --help)
  towncrier_checker.py --version
  towncrier_checker.py  --token=<token> --beginning=<tag> --repo=<repo>

Options:
  -h --help          Show this screen.
  --version          Show version.
  --beginning=<tag>  Where in the History to begin
  --repo=<repo>      Which repository to look at on GitHub
  --token=<token>    The GitHub token to talk to the API

Package requirements:

- pygithub
- docopt
- gitpython

Can be installed with:

    pip install pygithub docopt gitpython

"""

import re
import os.path
from pathlib import Path

from git import Repo
from docopt import docopt
from github import Github


if __name__ == "__main__":
    arguments = docopt(__doc__, version="1.0")
    beginning = arguments["--beginning"]
    target_ghrepo = arguments["--repo"]
    github_token = arguments["--token"]
    ghrepo = Github(github_token).get_repo(target_ghrepo)
    repo = Repo(".")
    all_commits = [x for x in repo.iter_commits(f"{beginning}..HEAD")]
    merge_commits = [
        x for x in all_commits if "Merge pull request" in x.message
    ]
    prmatch = re.compile(f"^Merge pull request #([0-9]+) from.*")
    prs = {}
    authors = set()
    for x in merge_commits:
        match = prmatch.match(x.message)
        if match:
            issue_id = match.groups()[0]
            prs[issue_id] = "%s" % (x.message.splitlines()[2])

    # Find towncrier files and remove them from the PR list
    base = Path("docs/upcoming_changes")
    for file in base.glob("*.*.rst"):
        pr = str(file.relative_to(base)).split(".", 1)[0]
        if pr in prs:
            del prs[pr]
        else:
            print(f"No matching merge commit for {pr}")

    # Find PRs that are mentioned in CHANGE_LOG already
    # (most likely cherrypicked to previous bugfix releases)
    basepath = Path(os.path.dirname(__file__)) / ".."
    oldchangelog = basepath / "CHANGE_LOG"
    newchangelogs = (basepath / "docs" / "source" / "release").glob("*.rst")
    for path in [oldchangelog, *newchangelogs]:
        print('Checking', path)
        with open(path) as fin:
            # Use regex to find all PR numbers in the file and remove them
            pr_re = re.compile(r"\#(\d+)")
            prs_in_changelog = [
                match.group(1) for match in pr_re.finditer(fin.read())
            ]
            to_remove = {pr for pr in prs if pr in prs_in_changelog}
            for pr in to_remove:
                print(f"Removing {pr} as already in CHANGELOG")
                del prs[pr]

    # Print PRs with missing notes
    if prs:
        print("PRs that are missing towncrier notes and not skipped")
        for issue_id in sorted(prs):
            pr = ghrepo.get_pull(int(issue_id))
            if any("skip_release_notes" in label.name for label in pr.labels):
                continue  # skip
            else:
                print(issue_id, pr.html_url)
