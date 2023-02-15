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
        ref = "https://www.github.com/numba/numba/blob/main/" + text
    elif path.isdir(full_path):
        ref = "https://www.github.com/numba/numba/tree/main/" + text
    else:
        logger.warn("Failed to find file in repomap: " + text)
        ref = "https://www.github.com/numba/numba"
    return ref


def intersperse(lst, item):
    """ Insert item between each item in lst.

    Copied under CC-BY-SA from stackoverflow at:

    https://stackoverflow.com/questions/5920643/
    add-an-item-between-each-item-already-in-the-list

    """
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def ghfile_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """ Emit hyperlink nodes for a given file in repomap. """
    my_nodes = []
    if "{" in text:  # myfile.{c,h} - make two nodes
        # could have used regexes, but this will be fine..
        base = text[:text.find(".") + 1]
        exts = text[text.find("{") + 1:text.find("}")].split(",")
        for e in exts:
            node = nodes.reference(rawtext,
                                   base + e,
                                   refuri=make_ref(base + e),
                                   **options)
            my_nodes.append(node)
    elif "*" in text:  # path/*_files.py - link to directory
        # Could have used something from os.path, but this will be fine..
        ref = path.dirname(text) + path.sep
        node = nodes.reference(rawtext, text, refuri=make_ref(ref), **options)
        my_nodes.append(node)
    else:  # everything else is taken verbatim
        node = nodes.reference(rawtext, text, refuri=make_ref(text), **options)
        my_nodes.append(node)

    # insert separators if needed
    if len(my_nodes) > 1:
        my_nodes = intersperse(my_nodes, nodes.Text(" | "))
    return my_nodes, []


def setup(app):
    logger.info('Initializing ghfiles plugin')
    app.add_role('ghfile', ghfile_role)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
