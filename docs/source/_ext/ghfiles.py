import os.path as path
import subprocess
import shlex
from sphinx.util import logging
from docutils import nodes
logger = logging.getLogger(__name__)


# use an old git trick, to get the top-level, could have used ../ etc.. but
# this will be fine..
top = subprocess.check_output(shlex.split(
    "git rev-parse --show-toplevel")).strip().decode("utf-8")


def make_ref(text):
    """ Make hyperlink to Github """
    full_path = path.join(top, text)
    if path.isfile(full_path):
        ref = "https://github.com/numba/numba/blob/master/" + text
    elif path.isdir(full_path):
        ref = "https://www.github.com/numba/numba/tree/master/" + text
    else:
        print("Failed to find file:" + text)
        ref = "https://www.github.com/numba/numba"
    return ref


def intersperse(lst, item):
    """ Insert item between each item in lst. """
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def ghfile_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    my_nodes = []
    if "{" in text:  # myfile.{c, h} - make two nodes
        base = text[:text.find(".") + 1]
        exts = text[text.find("{") + 1:text.find("}")].split(",")
        for e in exts:
            node = nodes.reference(rawtext, base + e, refuri=make_ref(base + e), **options)
            my_nodes.append(node)
    elif "*" in text:  # path/*_files.py - link to directory
        ref = text[:text.rfind("/") + 1]
        node = nodes.reference(rawtext, text, refuri=make_ref(ref), **options)
        my_nodes.append(node)
    else:  # everything else is taken verbatim
        node = nodes.reference(rawtext, text, refuri=make_ref(text), **options)
        my_nodes.append(node)

    # insert seperators if needed
    if len(my_nodes) > 1:
        my_nodes = intersperse(my_nodes, nodes.Text(" | "))
    return my_nodes, []


def setup(app):
    logger.info('Initializing ghfiles plugin')
    app.add_role('ghfile', ghfile_role)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
