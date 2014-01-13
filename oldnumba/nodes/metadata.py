# -*- coding: utf-8 -*-
"""
Allow annotating AST nodes with some metadata, and querying for that metadata.
"""
from __future__ import print_function, division, absolute_import

import weakref

def create_metadata_env():
    return weakref.WeakKeyDictionary()

def annotate(env, node, **flags):
    func_env = env.translation.crnt
    assert func_env is not None
    if node not in func_env.ast_metadata:
        metadata = {}
        func_env.ast_metadata[node] = metadata
    else:
        metadata = func_env.ast_metadata[node]

    metadata.update(flags)

def query(env, node, key, default=None):
    func_env = env.translation.crnt
    assert func_env is not None

    node_metadata = func_env.ast_metadata.get(node, {})
    return node_metadata.get(key, default)
