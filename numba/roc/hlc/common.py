"""
Shared code for the low level compiler tooling
"""

from __future__ import print_function, division, absolute_import

from abc import abstractmethod, ABCMeta
from numba import six
import re

# These are for parsing labels and metadata
_re_labelname = re.compile(r"\n\.([0-9a-z_\.]+):", re.I) # label: .<stuff>:
_re_regname = re.compile(r"%\.([0-9a-z_]+)", re.I) # register: %.<stuff>
_re_metadata_def = re.compile(r"\!\d+\s*=")
_re_metadata_correct_usage = re.compile(r"metadata\s*\![{'\"]")
_re_metadata_ref = re.compile(r"\!\d+")

# These are for parsing alloca instructions
_re_alloca_quoted = re.compile('(.*)"(\$.*)".*')
_re_alloca_parts = re.compile('(.*)=(.*alloca(.*))')

def add_metadata_type(ir):
    """
    Rewrite metadata since llvm3.6 dropped the "metadata" type prefix.
    """
    buf = []
    for line in ir.splitlines():
        # If the line is a metadata
        if _re_metadata_def.match(line):
            # Does not contain any correct usage (Maybe already fixed)
            if None is _re_metadata_correct_usage.search(line):
                line = line.replace('!{', 'metadata !{')
                line = line.replace('!"', 'metadata !"')

                def sub_metadata(m):
                    return "metadata {0}".format(m.group(0))

                line = _re_metadata_ref.sub(sub_metadata, line)
                line = line.lstrip('metadata ')
        buf.append(line)

    return '\n'.join(buf)


def rename_register(llvmir):
    """
    HLC does not like variable with '.' prefix.
    """
    def repl(mat):
        return '%_dot_.{0}'.format(mat.group(1))

    return _re_regname.sub(repl, llvmir)


def rename_label(llvmir):
    """
    HLC does not like a label with '.' prefix.
    """
    def repl(mat):
        return '_dot_.{0}:'.format(mat.group(1))

    return _re_labelname.sub(repl, llvmir)


def adapt_llvm_version(llvmir):
    """
    Adapt the LLVM IR to match the syntax required by HLC.
    """
    llvmir = rename_register(llvmir)
    llvmir = rename_label(llvmir)
    #   return add_metadata_type(llvmir)
    return llvmir


def alloca_addrspace_correction(llvmir):
    """
    rewrites llvmir such that alloca's go into addrspace(5) and are then
    addrspacecast back to to addrspace(0). Alloca into 5 is a requirement of
    the datalayout specification.
    """
    lines = llvmir.splitlines()
    mangle = '__tmp'
    new_ir = []
    for l in lines:
        # pluck lines containing alloca
        if 'alloca' in l:
            assignee, alloca_match, ptrty = _re_alloca_parts.match(l).groups()
            q_match = _re_alloca_quoted.match(assignee)
            if q_match:
                start, var = q_match.groups()
                var = var.strip()
                name_fmt = '%s"%s"'
                old_name = name_fmt % (start, var)
                new_name = name_fmt % (start, var + mangle)
            else:
                old_name = assignee.strip()
                new_name = old_name + mangle
            allocaline = "%s = %s, addrspace(5)" % (new_name, alloca_match)
            castline_fmt = ("%s = addrspacecast %s addrspace(5)* "
                            "%s to %s addrspace(0)*")
            castline =  castline_fmt % (old_name, ptrty, new_name, ptrty)
            new_ir.append(allocaline)
            new_ir.append(castline)
        else:
            new_ir.append(l)
    return '\n'.join(new_ir)


@six.add_metaclass(ABCMeta)
class _AMDGCNModule(object):
    """
    The AMDCGN LLVM module contract
    """

    @abstractmethod
    def load_llvm(self, llvmir):
        pass

    @abstractmethod
    def link_builtins(self, main):
        pass

    @abstractmethod
    def generateGCN(self, llvmir):
        pass


class AMDGCNModule(object):
    """
    The AMDCGN LLVM module contract
    """

    bitcodes = [
    "opencl.amdgcn.bc",
    "ocml.amdgcn.bc",
    "ockl.amdgcn.bc",
    "oclc_correctly_rounded_sqrt_off.amdgcn.bc",
    "oclc_daz_opt_off.amdgcn.bc",
    "oclc_finite_only_off.amdgcn.bc",
    "oclc_isa_version_803.amdgcn.bc",
    "oclc_unsafe_math_off.amdgcn.bc",
    "irif.amdgcn.bc"
    ]

    def __init__(self):
        self._finalized = False

    def _preprocess(self, llvmir):
        version_adapted = adapt_llvm_version(llvmir)
        alloca_fixed = alloca_addrspace_correction(version_adapted)
        return alloca_fixed

    def load_llvm(self, llvmir):
        pass

    def link_builtins(self, main):
        pass

    def generateGCN(self):
        pass

