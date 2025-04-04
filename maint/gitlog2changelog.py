"""gitlog2changelog.py

Usage:
  gitlog2changelog.py (-h | --help)
  gitlog2changelog.py --version
  gitlog2changelog.py --token=<token> --beginning=<tag> --repo=<repo> [--summary]

Options:
  -h --help          Show this screen.
  --version          Show version.
  --beginning=<tag>  Where in the History to begin
  --repo=<repo>      Which repository to look at on GitHub
  --token=<token>    The GitHub token to talk to the API
  --summary          Show total count for each section

"""

import re

from git import Repo
from docopt import docopt
from github import Github


ghrepo = None


def get_pr(pr_number):
    return ghrepo.get_pull(pr_number)


def hyperlink_user(user_obj):
    return "`%s <%s>`_" % (user_obj.login, user_obj.html_url)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='1.0')
    beginning = arguments['--beginning']
    target_ghrepo = arguments['--repo']
    github_token = arguments['--token']
    summary = arguments["--summary"]
    ghrepo = Github(github_token).get_repo(target_ghrepo)
    repo = Repo('.')
    all_commits = [x for x in repo.iter_commits(f'{beginning}..HEAD')]
    merge_commits = [x for x in all_commits
                     if 'Merge pull request' in x.message]
    prmatch = re.compile(
        r'^Merge pull request #([0-9]+) from.*')
    ordered = {}
    authors = set()
    for x in merge_commits:
        match = prmatch.match(x.message)
        if match:
            issue_id = match.groups()[0]
            ordered[issue_id] = None

    print("Pull-Requests:\n")
    missing_authors = set()
    for k in sorted(ordered.keys(), key=int):
        pull = get_pr(int(k))
        hyperlink = "`#%s <%s>`_" % (k, pull.html_url)
        # get all users for all commits
        pr_authors = set()
        for c in pull.get_commits():
            author = c.author
            if author:
                pr_authors.add(c.author)
            else:
                missing_authors.add((pull, c))
            if c.committer and c.committer.login != "web-flow":
                pr_authors.add(c.committer)
            elif not author:
                missing_authors.add((pull, c))
        
        pr_title = pull.title
        print("* PR %s: %s (%s)" % (hyperlink, pr_title,
                                    " ".join([hyperlink_user(u) + '_' for u in
                                              pr_authors])))
        for a in pr_authors:
            authors.add(a)
    if missing_authors:
        print("\n===========================WARNING=================================\n")
        print("Following PR commits are missing authors and may need manual changes:\n")
        for pull, commit in missing_authors:
            print(f"* {commit} in PR #{pull.number}: {pull.title}")
        print("\n===================================================================\n")
    if summary:
        print("\nTotal PRs: %s\n" % len(ordered))
    else:
        print()
    print("Authors:\n")
    [print('* %s' % hyperlink_user(x)) for x in sorted(authors, key=lambda x:
                                                       x.login.lower())]
    if summary:
        print("\nTotal authors: %s" % len(authors))
