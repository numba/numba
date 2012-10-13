"""
Code writers and formatters. Subclass CodeWriter to suit the needs of
a certain code generator backend.
"""

try:
    from Cython.Compiler import Tempita as tempita
except ImportError:
    try:
        import tempita
    except ImportError:
        tempita = None

class CodeWriter(object):
    """
    Write code as objects for later assembly.

    .. attribute:: loop_levels

        CodeWriter objects just before the start of each loop

    .. attribute:: tiled_loop_levels

        same as loop_levels, but takes into account tiled loop patterns

    .. attribute:: cleanup_levels

        CodeWriter objects just after the end of each loop

    .. attribute:: declaration_levels

        same as loop_levels, but a valid insertion point for C89 declarations
    """

    error_handler = None

    def __init__(self, context, buffer=None):
        self.buffer = buffer or _CodeTree()
        self.context = context

        self.loop_levels = []
        self.tiled_loop_levels = []
        self.declaration_levels = []

    @classmethod
    def clone(cls, other, context, buffer):
        return cls(context, buffer)

    def insertion_point(self):
        """
        Create an insertion point for the code writer. Any code written
        to this insertion point (later on) is inserted in the output code at
        the point where this method was called.
        """
        result = self.clone(self, self.context, self.buffer.insertion_point())
        result.loop_levels = list(self.loop_levels)
        result.tiled_loop_levels = list(self.tiled_loop_levels)
        result.declaration_levels = list(self.declaration_levels)
        return result

    def write(self, value):
        self.buffer.output.append(value)

    def put_label(self, label):
        "Insert a label in the code"
        self.write(label)

    def put_goto(self, label):
        "Jump to a label. Implement in subclasses"

class CCodeWriter(CodeWriter):
    """
    Code writer to write C code. Has both a prototype buffer and an
    implementation buffer. The prototype buffer will contain the C
    prototypes, and the implementation buffer the actual function
    code.
    """

    def __init__(self, context, buffer=None, proto_code=None):
        super(CCodeWriter, self).__init__(context, buffer)
        if proto_code is None:
            self.proto_code = type(self)(context, proto_code=False)
        self.indent = 0

    def put_label(self, label):
        "Insert a C label"
        self.putln('%s:' % self.mangle(label.name))

    def put_goto(self, label):
        "Jump to (goto) a label"
        self.putln("goto %s;" % self.mangle(label.name))

    def putln(self, s):
        "Write a code string as a line. Also performs indentation"
        self.indent -= s.count('}')
        self.write("%s%s\n" % (self.indent * '    ', s))
        self.indent += s.count('{')

    def mangle(self, s):
        "Mangle symbol names"
        return "__mini_mangle_%s" % s

    @classmethod
    def clone(cls, other, context, buffer):
        result = super(CCodeWriter, cls).clone(other, context, buffer)
        result.indent = other.indent
        return result

def sub_tempita(s, context, file=None, name=None):
    "Run the tempita template engine on string the given string."
    if not s:
        return None

    if file:
        context['__name'] = "%s:%s" % (file, name)
    elif name:
        context['__name'] = name

    if tempita is None:
        raise RuntimeError("Tempita was not installed")

    return tempita.sub(s, **context)

class TempitaCodeWriter(CodeWriter):
    """
    Code writer which supports writing Tempita strings. See
    http://pythonpaste.org/tempita/ for documentation on Tempita.
    """

    def putln(self, string, context_dict):
        self.write(sub_tempita(string) + '\n')

class CodeFormatter(object):
    """
    Default code formatting, which returns the formatted code as a list
    of objects (the ones written to the :py:class:`minivect.codegen.CodeWriter`)
    """
    def format(self, codewriter):
        return codewriter.buffer.getvalue()

class CodeStringFormatter(CodeFormatter):
    "Format code as strings"
    def format(self, codewriter):
        return "".join(codewriter.buffer.getvalue())

class CCodeStringFormatter(CodeStringFormatter):
    "Format the prototype and code implementation"

    def format(self, codewriter):
        return ("".join(codewriter.proto_code.buffer.getvalue()),
                "".join(codewriter.buffer.getvalue()))

class _CodeTree(object):
    """
    See Cython/StringIOTree
    """

    def __init__(self, output=None, condition=None):
        self.prepended_children = []
        self.output = output or []

    def _getvalue(self, result):
        for child in self.prepended_children:
            child._getvalue(result)
        result.extend(self.output)

    def getvalue(self):
        result = []
        self._getvalue(result)
        return result

    def clone(self, output=None):
        return type(self)(output)

    def commit(self):
        if self.output:
            self.prepended_children.append(self.clone(self.output))
            self.output = []

    def insertion_point(self):
        self.commit()
        ip = self.clone()
        self.prepended_children.append(ip)
        return ip