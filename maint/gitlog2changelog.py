"""gitlog2changelog.py

Usage:
  gitlog2changelog.py (-h | --help)
  gitlog2changelog.py --version
  gitlog2changelog.py --token=<token> --beginning=<tag> --repo=<repo> \
          --digits=<digits> [--summary]

Options:
  -h --help          Show this screen.
  --version          Show version.
  --beginning=<tag>  Where in the History to begin
  --repo=<repo>      Which repository to look at on GitHub
  --token=<token>    The GitHub token to talk to the API
  --digits=<digits>  The number of digits to use in the issue finding regex
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
    regex_digits = arguments['--digits']
    summary = arguments["--summary"]
    ghrepo = Github(github_token).get_repo(target_ghrepo)
    repo = Repo('.')
    all_commits = [x for x in repo.iter_commits(f'{beginning}..HEAD')]
    merge_commits = [x for x in all_commits
                     if 'Merge pull request' in x.message]
    prmatch = re.compile(
        f'^Merge pull request #([0-9]{{{regex_digits}}}) from.*')
    ordered = {}
    authors = set()
    for x in merge_commits:
        match = prmatch.match(x.message)
        if match:
            issue_id = match.groups()[0]
            ordered[issue_id] = "%s" % (x.message.splitlines()[2])
    print("Pull-Requests:\n")
    for k in sorted(ordered.keys()):
        pull = get_pr(int(k))
        hyperlink = "`#%s <%s>`_" % (k, pull.html_url)
        # get all users for all commits
        pr_authors = set()
        for c in pull.get_commits():
            if c.author:
                pr_authors.add(c.author)
            if c.committer and c.committer.login != "web-flow":
                pr_authors.add(c.committer)
        print("* PR %s: %s (%s)" % (hyperlink, ordered[k],
                                    " ".join([hyperlink_user(u) for u in
                                              pr_authors])))
        for a in pr_authors:
            authors.add(a)
    if summary:
        print("\nTotal PRs: %s\n" % len(ordered))
    else:
        print()
    print("Authors:\n")
    [print('* %s' % hyperlink_user(x)) for x in sorted(authors, key=lambda x:
                                                       x.login.lower())]
    if summary:
        print("\nTotal authors: %s" % len(authors))
