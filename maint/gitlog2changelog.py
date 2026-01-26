"""gitlog2changelog.py

Usage:
  gitlog2changelog.py (-h | --help)
  gitlog2changelog.py --version
  gitlog2changelog.py --token=<token> --beginning=<tag> --repo=<repo>

Options:
  -h --help          Show this screen.
  --version          Show version.
  --beginning=<tag>  Where in the History to begin
  --repo=<repo>      Which repository to look at on GitHub
  --token=<token>    The GitHub token to talk to the API

Dependencies:
  * docopt
  * pygithub
  * gitpython

"""

import re

from git import Repo
from docopt import docopt
from github import Github, Auth


def hyperlink_user(user_obj):
    try:
        return "`%s <%s>`_" % (user_obj.login, user_obj.html_url)
    except AttributeError:
        return user_obj


def author_key(x):
    return x.login.lower() if not isinstance(x, str) else x.lower()


if __name__ == '__main__':

    # Process command line arguments.
    arguments = docopt(__doc__, version='1.0')
    beginning = arguments['--beginning']
    target_ghrepo = arguments['--repo']
    github_token = arguments['--token']

    # this is an object representation of the repo on github using pygithub
    ghrepo = Github(auth=Auth.Token(github_token)).get_repo(target_ghrepo)
    # this is an object representation of the local git repo using gitpython
    repo = Repo('.')

    # # We obtain the merge commits for the given range by looking at the
    # commit message. If it begins with the string "Merge pull request" it is
    # either a true merge commit (with two parents) that originats from merging
    # a pull-requets OR it is a cherry-picked merge-commit. Both are fine and
    # should be included.
    merge_commits = [x for x in repo.iter_commits(f'{beginning}..HEAD')
                   if x.message.startswith('Merge pull request')]
    # We use the following regular expression to extract the GitHub ID of the
    # pull-request from the message.
    prmatch = re.compile(
        f'^Merge pull request #(\\d+) from.*')

    # Then, using the list, we use the regular expression above to extract the
    # ID of the pull-requests and build a dictionary mapping the Pull-Request
    # ID to it's description.
    pull_requests = {}
    for m in merge_commits:
        match = prmatch.match(m.message)
        number = int(match.group(1))
        # There are very rare cases where the commit message of the merge
        # commit created when merging a pull-request on GitHub does not include
        # the description. In that case, we use the GitHub API to obtain that
        # description.
        try:
            description = m.message.splitlines()[2]
        except IndexError:
            p = ghrepo.get_pull(number)
            description = p.title
        # Lastly, add the pull-request to the dictionary.
        pull_requests[number] = str(description)

    # Now that we have the pull-requests, we iterate through them to determine
    # the authors involved.
    authors = set()
    print("Pull-Requests:\n")
    for k in sorted(pull_requests.keys()):
        pr = ghrepo.get_pull(k)
        hyperlink = "`#%s <%s>`_" % (k, pr.html_url)
        # get all users for all commits
        pr_authors = set()
        for c in pr.get_commits():
            # Here we need to try to get some kind of an "author" string for
            # the given commit. The first thing we try is to get the GitHub
            # username associated with the email address in the commit
            # metadata. If this fails, when someone has used a 'user.email'
            # setting that does not correspond to a known account on GitHub, we
            # attempt to get the `user.name` setting from the commit directly.
            # We need to convert the commit from its pygithub style object to
            # its gitpython style object in order to access that information.
            # We do the same for the committer.
            author = (c.author if c.author is not None
                      else repo.commit(str(c.sha)).author.name)
            assert author is not None and author != ""
            pr_authors.add(author)
            committer = (c.committer 
                         if (c.committer is not None
                             and c.committer.login != "web-flow")
                         else repo.commit(str(c.sha)).committer.name)
            if committer != "GitHub":
                pr_authors.add(committer)
        print("* PR %s: %s (%s)" % (hyperlink, pull_requests[k],
                                    " ".join([hyperlink_user(u) for u in
                                              sorted(pr_authors, key=author_key)])))
        for a in pr_authors:
            authors.add(a)

    print("")
    print("Authors:\n")
    [print('* %s' % hyperlink_user(x)) for x in sorted(authors, key=author_key)]
