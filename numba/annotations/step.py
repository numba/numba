"""A light and fast template engine."""

import re


class Template(object):

    COMPILED_TEMPLATES = {} # {template string: code object, }
    # Regex for stripping all leading, trailing and interleaving whitespace.
    RE_STRIP = re.compile("(^[ \t]+|[ \t]+$|(?<=[ \t])[ \t]+|\A[\r\n]+|[ \t\r\n]+\Z)", re.M)

    def __init__(self, template, strip=True):
        """Initialize class"""
        super(Template, self).__init__()
        self.template = template
        self.options  = {"strip": strip}
        self.builtins = {"escape": lambda s: escape_html(s),
                         "setopt": lambda k, v: self.options.update({k: v}), }
        if template in Template.COMPILED_TEMPLATES:
            self.code = Template.COMPILED_TEMPLATES[template]
        else:
            self.code = self._process(self._preprocess(self.template))
            Template.COMPILED_TEMPLATES[template] = self.code

    def expand(self, namespace={}, **kw):
        """Return the expanded template string"""
        output = []
        namespace.update(kw, **self.builtins)
        namespace["echo"]  = lambda s: output.append(s)
        namespace["isdef"] = lambda v: v in namespace

        eval(compile(self.code, "<string>", "exec"), namespace)
        return self._postprocess("".join(output))

    def stream(self, buffer, namespace={}, **kw):
        """Expand the template and stream it to a file-like buffer."""

        def write_buffer(s, flush=False, cache = [""]):
            # Cache output as a single string and write to buffer.
            cache[0] += str(s)
            if flush and cache[0] or len(cache[0]) > 65536:
                buffer.write(cache[0])
                cache[0] = ""

        namespace.update(kw, **self.builtins)
        namespace["echo"]  = write_buffer
        namespace["isdef"] = lambda v: v in namespace

        eval(compile(self.code, "<string>", "exec"), namespace)
        write_buffer("", flush=True) # Flush any last cached bytes

    def _preprocess(self, template):
        """Modify template string before code conversion"""
        # Replace inline ('%') blocks for easier parsing
        o = re.compile("(?m)^[ \t]*%((if|for|while|try).+:)")
        c = re.compile("(?m)^[ \t]*%(((else|elif|except|finally).*:)|(end\w+))")
        template = c.sub(r"<%:\g<1>%>", o.sub(r"<%\g<1>%>", template))

        # Replace ({{x}}) variables with '<%echo(x)%>'
        v = re.compile("\{\{(.*?)\}\}")
        template = v.sub(r"<%echo(\g<1>)%>\n", template)

        return template

    def _process(self, template):
        """Return the code generated from the template string"""
        code_blk = re.compile(r"<%(.*?)%>\n?", re.DOTALL)
        indent = 0
        code = []
        for n, blk in enumerate(code_blk.split(template)):
            # Replace '<\%' and '%\>' escapes
            blk = re.sub(r"<\\%", "<%", re.sub(r"%\\>", "%>", blk))
            # Unescape '%{}' characters
            blk = re.sub(r"\\(%|{|})", "\g<1>", blk)

            if not (n % 2):
                # Escape double-quote characters
                blk = re.sub(r"\"", "\\\"", blk)
                blk = (" " * (indent*4)) + 'echo("""{0}""")'.format(blk)
            else:
                blk = blk.rstrip()
                if blk.lstrip().startswith(":"):
                    if not indent:
                        err = "unexpected block ending"
                        raise SyntaxError("Line {0}: {1}".format(n, err))
                    indent -= 1
                    if blk.startswith(":end"):
                        continue
                    blk = blk.lstrip()[1:]

                blk = re.sub("(?m)^", " " * (indent * 4), blk)
                if blk.endswith(":"):
                    indent += 1

            code.append(blk)

        if indent:
            err = "Reached EOF before closing block"
            raise EOFError("Line {0}: {1}".format(n, err))

        return "\n".join(code)

    def _postprocess(self, output):
        """Modify output string after variables and code evaluation"""
        if self.options["strip"]:
            output = Template.RE_STRIP.sub("", output)
        return output


def escape_html(x):
    """Escape HTML special characters &<> and quotes "'."""
    CHARS, ENTITIES = "&<>\"'", ["&amp;", "&lt;", "&gt;", "&quot;", "&#39;"]
    string = x if isinstance(x, basestring) else str(x)
    for c, e in zip(CHARS, ENTITIES): string = string.replace(c, e)
    return string
