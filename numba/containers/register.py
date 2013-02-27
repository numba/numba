from functools import partial

from numba.containers import typedlist
from numba.containers import typedtuple
from numba.containers import orderedcontainer
from numba.type_inference.module_type_inference import register_inferer

#-----------------------------------------------------------------------
# Register type function for typedlist construction
#-----------------------------------------------------------------------

def infer_tlist(type_node, iterable_node):
    return orderedcontainer.typedcontainer_infer(
        typedlist.compile_typedlist, type_node, iterable_node)

register_inferer(typedlist, 'typedlist', infer_tlist, pass_in_types=False)

#-----------------------------------------------------------------------
# Register type function for typedtuple construction
#-----------------------------------------------------------------------

def infer_ttuple(type_node, iterable_node):
    return orderedcontainer.typedcontainer_infer(
        typedtuple.compile_typedtuple, type_node, iterable_node)

register_inferer(typedtuple, 'typedtuple', infer_ttuple, pass_in_types=False)
