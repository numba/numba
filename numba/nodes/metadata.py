"""
Allow annotating AST nodes with some metadata, and querying for that metadata.
"""

import weakref
import collections

def create_metadata_env():
    return collections.defaultdict(weakref.WeakKeyDictionary)

def annotate(env, node, key, value):
    func_env = env.translation.crnt
    assert func_env is not None
    func_env.ast_metadata[node][key] = value

def query(env, node, key):
    func_env = env.translation.crnt
    assert func_env is not None
    node_metadata = func_env.ast_metadata[node]
    return node_metadata[key]
