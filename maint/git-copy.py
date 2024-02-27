""" git-copy.py

Synopsis
--------

This script implements a two-way copy of files in git using
`git mv` to preserve history for `git blame`.

$ python git-copy.py FILES

Details
-------

For a given file -- say `myfile.py` -- we create three copies in three branches
using the branch `copied` as a base and target (which will also be created).

Branch `new_files` contains `new_myfile.py`
Branch `old_files` contains `old_myfile.py`
Branch `preserved_files` contains `preserved_myfile.py`

Then we use a git octo-merge to merge these three branches into the `copied`
branch, this results in three "copies" (two copies and one original).

Lastly, we remove the prefix `preserved_` again to restore the original
`myfile.py` in a final commit. This is a 'trick' used to restore and hence
retain the original `myfile.py`. And then the temporary branches are deleted
such that only the base branch `copied` remains.

If the script succeeds, the following files will be present:

new_myfile.py
old_myfile.py
myfile.py

.. and all three will have sane `git blame` output.

The script is rudimentary and doesn't fail gracefully or clean up in case it
fails. If something goes wrong, read the output of the error message closely.

Tip: A good way to undo the created branches if something does go wrong is to
use:

$ git checkout main
$ git branch -D copied old_files, new_files, preserved_files

... assuming `main` is your main development branch.

Lastly, You can hand over as many files as your shell and Python implementation
will handle. Hopefully this should suffice for all practical purposes.

Dependencies
------------

* https://pypi.org/project/sh/

See also
--------

This script was inspired by a technique mentioned in:

https://stackoverflow.com/questions/16937359/git-copy-file-preserving-history/44036771#44036771

"""

import os
import sys

import sh

BASE_BRANCH = "copied"
FILES = sys.argv[1:]

NEW_BRANCH_NAME = "new_files"
NEW_PREFIX = "new_"
OLD_BRANCH_NAME = "old_files"
OLD_PREFIX = "old_"
PRESERVED_BRANCH_NAME = "preserved_files"
PRESERVED_PREFIX = "preserved_"


def copy_files(branch: str, prefix: str, remove_prefix: bool = False) -> None:
    """ copy files on a branch using git mv

    This function will create a git branch and then copy all files in global
    FILES to insert or remove a prefix into or from the filename. Lastly, those
    changes will be committed to git.

    branch: the name of the branch
    prefix: the prefix to insert or remove
    remove_prefix: insert or remove

    """
    sh.git.checkout(BASE_BRANCH)
    if not remove_prefix:
        print(f"Setting up git branch: '{branch}'")
        sh.git.branch(branch)
        sh.git.checkout(branch)
    for file in FILES:
        head, tail = os.path.split(file)
        if remove_prefix:
            origin = os.path.join(head, f"{prefix}{tail}")
            print(f"git mv {origin} {file}")
            sh.git.mv(origin, file)
        else:
            move_target = os.path.join(head, f"{prefix}{tail}")
            print(f"git mv {file} {move_target}")
            sh.git.mv(file, move_target)
    if remove_prefix:
        print(f"Creating commit to remove prefix: '{prefix}'")
        sh.git.commit("--no-verify", "-m", f"remove prefix: '{prefix}'")
    else:
        print(f"Creating commit for prefix: '{prefix}'")
        sh.git.commit("--no-verify", "-m", f"copy files to prefix: '{prefix}'")


if __name__ == "__main__":

    # exit if no arguments given
    if len(FILES) == 0:
        print("usage: git-copy FILES")
        sys.exit(1)

    # setting up the `copied` branch
    print(f"Setting up git branch: '{BASE_BRANCH}'")
    sh.git.branch(BASE_BRANCH)
    sh.git.checkout(BASE_BRANCH)

    # create the three branches and do the git mv and commit
    copy_files(NEW_BRANCH_NAME, NEW_PREFIX)
    copy_files(OLD_BRANCH_NAME, OLD_PREFIX)
    copy_files(PRESERVED_BRANCH_NAME, PRESERVED_PREFIX)

    # perform the octo-merge of the three branches
    sh.git.checkout(BASE_BRANCH)
    sh.git.merge(NEW_BRANCH_NAME, OLD_BRANCH_NAME, PRESERVED_BRANCH_NAME)

    # restore the originals
    copy_files(PRESERVED_BRANCH_NAME, PRESERVED_PREFIX, remove_prefix=True)

    # remove stale branches
    sh.git.branch("-d", NEW_BRANCH_NAME, OLD_BRANCH_NAME, PRESERVED_BRANCH_NAME)

    print("...done")
