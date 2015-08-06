from __future__ import print_function, absolute_import

import re

re_regname = re.compile(r"%\.([0-9a-z_]+)", re.I)
re_metadata_def = re.compile(r"\!\d+\s*=")
re_metadata_correct_usage = re.compile(r"metadata\s*\![{'\"]")
re_metadata_ref = re.compile(r"\!\d+")


def add_metadata_type(ir):
    """
    Rewrite metadata since llvm3.6 dropped the "metadata" type prefix.
    """
    buf = []
    for line in ir.splitlines():
        # If the line is a metadata
        if re_metadata_def.match(line):
            # Does not contain any correct usage (Maybe already fixed)
            if None is re_metadata_correct_usage.search(line):
                line = line.replace('!{', 'metadata !{')
                line = line.replace('!"', 'metadata !"')

                def sub_metadata(m):
                    return "metadata {0}".format(m.group(0))

                line = re_metadata_ref.sub(sub_metadata, line)
                line = line.lstrip('metadata ')
        buf.append(line)

    return '\n'.join(buf)


def rename_register(llvmir):
    """
    HLC does not like variable with '.' prefix.
    """
    def repl(mat):
        return '%_dot_.{0}'.format(mat.group(1))

    return re_regname.sub(repl, llvmir)


def adapt_llvm_version(llvmir):
    """
    Adapt the LLVM IR to match the syntax required by HLC.
    """
    llvmir = rename_register(llvmir)
    return add_metadata_type(llvmir)
