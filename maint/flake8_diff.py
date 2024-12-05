"""
python flake8_diff.py [git_diff_compare]

Report flake8 errors on file-lines changed.
Uses `git diff --unified=0 <git_diff_compare>` to detect changes.
By default `<git_diff_compare>` is `main...`.
Uses `flake8` to detect style error and ignores excluded file in `.flake8`.
"""
import re
import sys
import subprocess as subp


def split_file(diff):
    # Split diff by:
    # +++ b/{path}
    pat = re.compile(r"\+\+\+ b/(?P<path>.+)")
    filename = None
    active = None
    for ln in diff.splitlines():
        m = pat.match(ln)
        new_filename = None
        if m is not None:
            new_filename = m.groupdict()["path"]

        if filename is None:
            filename = new_filename
            active = []
        elif new_filename:
            yield filename, active
            filename = None
            active = None
        else:
            active.append(ln)

    if filename is not None:
        assert active is not None
        yield filename, active


def get_changed_lines(diff):
    for path, lines in split_file(diff):
        # @@ -{neg_line}[,{count}] +{pos_line}[,{count}] @@
        pat = re.compile(
            r"@@ -\d+(,\d+)? \+(?P<pos_line>\d+)(,(?P<pos_count>\d+))? @@"
        )

        include_lines = set()
        for ln in lines:
            m = pat.match(ln)
            if m is not None:
                gd = m.groupdict()
                pos_line = int(gd["pos_line"])
                pos_count = int(gd.get("pos_count") or 1)
                include_lines.update(range(pos_line, pos_line + pos_count))

        yield path, include_lines


def find_diff_lines(compare):
    cmd = ["git", "diff", "--unified=0", compare]
    out = subp.check_output(cmd)
    diff = out.decode()
    return get_changed_lines(diff)


def main(git_diff_compare="main..."):
    pat_line_col = re.compile(r"[^:]+:(\d+):\d+")
    for path, lines_changed in find_diff_lines(git_diff_compare):
        # Use --exclude to disable file exclusion patterns in .flake8
        proc = subp.run(["flake8", "--exclude=__pycache__", path],
                        stdout=subp.PIPE)
        if proc.returncode != 0:
            for ln in proc.stdout.decode().splitlines():
                m = pat_line_col.match(ln)
                if m is not None:
                    line = int(m.group(1))
                    if line in lines_changed:
                        print(ln)


if __name__ == "__main__":
    main(*sys.argv[1:])
