# -*- coding: utf-8 -*-
"""
Module that deals with NumPy array slicing.

    - normalize ellipses
    - recognize newaxes
    - track how contiguity is affected (C or Fortran)
"""
from __future__ import print_function, division, absolute_import

import ast

from numba import *
from numba import nodes, typesystem
from numba.symtab import Variable

def unellipsify(node, slices, subscript_node):
    """
    Given an array node `node`, process all AST slices and create the
    final type:

        - process newaxes (None or numpy.newaxis)
        - replace Ellipsis with a bunch of ast.Slice objects
        - process integer indices
        - append any missing slices in trailing dimensions
    """
    type = node.variable.type

    if not type.is_array:
        assert type.is_object
        return object_, node

    if (len(slices) == 1 and nodes.is_constant_index(slices[0]) and
            slices[0].value.pyval is Ellipsis):
        # A[...]
        return type, node

    result = []
    seen_ellipsis = False

    # Filter out newaxes
    newaxes = [newaxis for newaxis in slices if nodes.is_newaxis(newaxis)]
    n_indices = len(slices) - len(newaxes)

    full_slice = ast.Slice(lower=None, upper=None, step=None)
    full_slice.variable = Variable(typesystem.slice_)
    ast.copy_location(full_slice, slices[0])

    # process ellipses and count integer indices
    indices_seen = 0
    for slice_node in slices[::-1]:
        slice_type = slice_node.variable.type
        if slice_type.is_ellipsis:
            if seen_ellipsis:
                result.append(full_slice)
            else:
                nslices = type.ndim - n_indices + 1
                result.extend([full_slice] * nslices)
                seen_ellipsis = True
        elif (slice_type.is_slice or slice_type.is_int or
              nodes.is_newaxis(slice_node)):
            indices_seen += slice_type.is_int
            result.append(slice_node)
        else:
            # TODO: Coerce all object operands to integer indices?
            # TODO: (This will break indexing with the Ellipsis object or
            # TODO:  with slice objects that we couldn't infer)
            return object_, nodes.CoercionNode(node, object_)

    # Reverse our reversed processed list of slices
    result.reverse()

    # append any missing slices (e.g. a2d[:]
    result_length = len(result) - len(newaxes)
    if result_length < type.ndim:
        nslices = type.ndim - result_length
        result.extend([full_slice] * nslices)

    subscript_node.slice = ast.ExtSlice(result)
    ast.copy_location(subscript_node.slice, slices[0])

    # create the final array type and set it in value.variable
    result_dtype = node.variable.type.dtype
    result_ndim = node.variable.type.ndim + len(newaxes) - indices_seen
    if result_ndim > 0:
        result_type = result_dtype[(slice(None),) * result_ndim]
    elif result_ndim == 0:
        result_type = result_dtype
    else:
        result_type = object_

    return result_type, node
