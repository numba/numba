"""
Provide threadsafe access to LLVM.
"""


from __future__ import absolute_import
import threading
import llvmlite.binding as ll


_llvm_lock = threading.Lock()


def _parse_assembly_threadsafe(mod):
    """
    Threadsafe version of ``llvmlite.binding.parse_assembly``.
    """
    with _llvm_lock:
        return ll.parse_assembly(mod)


def _link_module_threadsafe(mod1, mod2, preserve=False):
    """
    Threadsafe version of ``llvmlite.binding.Module.link_in``.
    """
    with _llvm_lock:
        mod1.link_in(mod2, preserve=preserve)
