#! /usr/bin/env python

import sys
import os

from github3 import login
import github3


def fetch(orgname, reponame, last_num, gh):

    repo = gh.repository(orgname, reponame)
    issues = repo.issues(state='all')

    opened_issues = []
    closed_issues = []

    opened_prs = []
    closed_prs = []

    max_iss_num = 0
    for issue in issues:
        info = issue.as_dict()
        iss_num = int(info['number'])
        max_iss_num = max(max_iss_num, iss_num)

        if iss_num <= last_num:
            break
        merged = False
        if issue.pull_request_urls:
            # Is PR?
            merged = bool(info['pull_request'].get("merged_at"))
            where = {'opened': opened_prs, 'closed': closed_prs}
        else:
            # Is Issues
            where = {'opened': opened_issues, 'closed': closed_issues}

        line = f"{' - merged ' if merged else ''}- [{reponame}"\
               f"#{info['number']}]({info['html_url']}) - {info['title']}"

        # Is issue already merged
        if issue.is_closed():
            where['closed'].append(line)
        else:
            where['opened'].append(line)

    return {
        'opened_issues': opened_issues,
        'closed_issues': closed_issues,
        'opened_prs': opened_prs,
        'closed_prs': closed_prs,
        'max_iss_num': max_iss_num,
    }


def display(data):
    print("## 1. New Issues")
    for line in reversed(data['opened_issues']):
        print(line)
    print()

    print("### Closed Issues")
    for line in reversed(data['closed_issues']):
        print(line)
    print()

    print("## 2. New PRs")
    for line in reversed(data['opened_prs']):
        print(line)
    print()

    print("### Closed PRs")
    for line in reversed(data['closed_prs']):
        print(line)
    print()


def main(numba_last_num, llvmlite_last_num, user=None, password=None):

    if user is not None and password is not None:
        gh = login(str(user), password=str(password))
    else:
        gh = github3

    numba_data = fetch("numba", "numba", numba_last_num, gh)
    llvmlite_data = fetch("numba", "llvmlite", llvmlite_last_num, gh)

    # combine data
    data = {
        'opened_issues':
            llvmlite_data['opened_issues'] +
            numba_data['opened_issues'],
        'closed_issues':
            llvmlite_data['closed_issues'] +
            numba_data['closed_issues'],
        'opened_prs':
            llvmlite_data['opened_prs'] +
            numba_data['opened_prs'],
        'closed_prs':
            llvmlite_data['closed_prs'] +
        numba_data['closed_prs'],
    }

    display(data)

    print(f"(last numba: {numba_data['max_iss_num']};"
          f"llvmlite {llvmlite_data['max_iss_num']})")


help_msg = """
Usage:
    {program_name} <numba_last_num> <llvmlite_last_num>
"""

if __name__ == '__main__':
    program_name = sys.argv[0]
    try:
        [numba_last_num, llvmlite_last_num] = sys.argv[1:]
    except ValueError:
        print(help_msg.format(program_name=program_name))
    else:
        main(int(numba_last_num),
             int(llvmlite_last_num),
             user=os.environ.get("GHUSER"),
             password=os.environ.get("GHPASS"))
