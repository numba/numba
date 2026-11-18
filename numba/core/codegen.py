import warnings
import functools
import locale
import os
import sys
import weakref
import ctypes
import html
import textwrap

import llvmlite.binding as ll
import llvmlite.ir as llvmir


# OrcJIT migration debugging: when NUMBA_ORCJIT_DEBUG=1, dump per-library
# pre-add symbol tables, post-add rename maps, and per-lookup hits/misses to
# stdout. Off by default; the helpers short-circuit when the flag is unset
# so production overhead is one env-var read at import.
_ORCJIT_DEBUG = bool(int(os.environ.get("NUMBA_ORCJIT_DEBUG", "0") or "0"))


def _orcjit_debug_dump_module_symbols(library_name, ll_module):
    if not _ORCJIT_DEBUG:
        return
    defs, decls = [], []
    for fn in ll_module.functions:
        bucket = decls if fn.is_declaration else defs
        bucket.append(("func", fn.linkage, fn.name))
    for gv in ll_module.global_variables:
        bucket = decls if gv.is_declaration else defs
        bucket.append(("gv", gv.linkage, gv.name))
    print(f"[DBG codegen] _add_module library={library_name!r} "
          f"defs={len(defs)} decls={len(decls)}", flush=True)
    for kind, link, name in defs:
        print(f"  def  {kind:4s} {str(link):14s} : {name}", flush=True)
    for kind, link, name in decls:
        print(f"  decl {kind:4s} {str(link):14s} : {name}", flush=True)
    sys.stdout.flush()


def _orcjit_debug_dump_rename_map(engine, jit_handle, ll_module):
    if not _ORCJIT_DEBUG:
        return
    try:
        rmap = (engine._ee._get_rename_map(jit_handle)
                if jit_handle is not None else {})
    except Exception as e:
        print(f"[DBG codegen] _get_rename_map failed: {e!r}", flush=True)
        return
    print(f"[DBG codegen] handle={jit_handle} rename map size={len(rmap)}",
          flush=True)
    for old, new in rmap.items():
        print(f"  rename      : {old}  ->  {new}", flush=True)
    pre_defined = [fn.name for fn in ll_module.functions
                   if not fn.is_declaration]
    pre_defined += [gv.name for gv in ll_module.global_variables
                    if not gv.is_declaration]
    not_renamed = [n for n in pre_defined if n not in rmap]
    print(f"[DBG codegen] not-renamed defined symbols ({len(not_renamed)}):",
          flush=True)
    for n in not_renamed:
        print(f"  preserved   : {n}", flush=True)


def _orcjit_debug_lookup(library_name, jit_handle, name, defined, addr=None):
    if not _ORCJIT_DEBUG:
        return
    print(f"[DBG codegen] get_pointer_to_function lib={library_name!r} "
          f"handle={jit_handle} name={name!r} defined={defined}", flush=True)
    if defined and addr is not None:
        print(f"[DBG codegen]   -> addr=0x{addr:x}", flush=True)


def _orcjit_debug_finalize(library_name, jit_handle, when):
    if not _ORCJIT_DEBUG:
        return
    print(f"[DBG codegen] finalize_object {when} lib={library_name!r} "
          f"handle={jit_handle}", flush=True)


def _orcjit_debug_resolve(pending, results):
    if not _ORCJIT_DEBUG or not pending:
        return
    print(f"[DBG codegen] RuntimeLinker.resolve pending={pending!r}",
          flush=True)
    for name, addr in results:
        print(f"[DBG codegen]   resolved {name!r} -> 0x{addr:x}", flush=True)

from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection


_x86arch = frozenset(['x86', 'i386', 'i486', 'i586', 'i686', 'i786',
                      'i886', 'i986'])


def _is_x86(triple):
    arch = triple.split('-')[0]
    return arch in _x86arch


def _parse_refprune_flags():
    """Parse refprune flags from the `config`.

    Invalid values are ignored an warn via a `NumbaInvalidConfigWarning`
    category.

    Returns
    -------
    flags : llvmlite.binding.RefPruneSubpasses
    """
    flags = config.LLVM_REFPRUNE_FLAGS.split(',')
    if not flags:
        return 0
    val = 0
    for item in flags:
        item = item.strip()
        try:
            val |= getattr(ll.RefPruneSubpasses, item.upper())
        except AttributeError:
            warnings.warn(f"invalid refprune flags {item!r}",
                          NumbaInvalidConfigWarning)
    return val


def dump(header, body, lang):
    if config.HIGHLIGHT_DUMPS:
        try:
            import pygments
        except ImportError:
            msg = "Please install pygments to see highlighted dumps"
            raise ValueError(msg)
        else:
            from pygments import highlight
            from pygments.lexers import GasLexer as gas_lexer
            from pygments.lexers import LlvmLexer as llvm_lexer
            from pygments.formatters import Terminal256Formatter
            from numba.misc.dump_style import by_colorscheme

            lexer_map = {'llvm': llvm_lexer, 'asm': gas_lexer}
            lexer = lexer_map[lang]
            def printer(arg):
                print(highlight(arg, lexer(),
                      Terminal256Formatter(style=by_colorscheme())))
    else:
        printer = print
    print('=' * 80)
    print(header.center(80, '-'))
    printer(body)
    print('=' * 80)


class _CFG(object):
    """
    Wraps the CFG graph for different display method.

    Instance of the class can be stringified (``__repr__`` is defined) to get
    the graph in DOT format.  The ``.display()`` method plots the graph in
    PDF.  If in IPython notebook, the returned image can be inlined.
    """
    def __init__(self, cres, name, py_func, **kwargs):
        self.cres = cres
        self.name = name
        self.py_func = py_func
        fn = cres.get_function(name)
        self.dot = ll.get_function_cfg(fn)
        self.kwargs = kwargs

    def pretty_printer(self, filename=None, view=None, render_format=None,
                       highlight=True,
                       interleave=False, strip_ir=False, show_key=True,
                       fontsize=10):
        """
        "Pretty" prints the DOT graph of the CFG.
        For explanation of the parameters see the docstring for
        numba.core.dispatcher::inspect_cfg.
        """
        import graphviz as gv
        import re
        import json
        import inspect
        from llvmlite import binding as ll
        from numba.typed import List
        from types import SimpleNamespace
        from collections import defaultdict

        _default = False
        _highlight = SimpleNamespace(incref=_default,
                                    decref=_default,
                                    returns=_default,
                                    raises=_default,
                                    meminfo=_default,
                                    branches=_default,
                                    llvm_intrin_calls=_default,
                                    function_calls=_default,)
        _interleave = SimpleNamespace(python=_default, lineinfo=_default)

        def parse_config(_config, kwarg):
            """ Parses the kwarg into a consistent format for use in configuring
            the Digraph rendering. _config is the configuration instance to
            update, kwarg is the kwarg on which to base the updates.
            """
            if isinstance(kwarg, bool):
                for attr in _config.__dict__:
                    setattr(_config, attr, kwarg)
            elif isinstance(kwarg, dict):
                for k, v in kwarg.items():
                    if k not in _config.__dict__:
                        raise ValueError("Unexpected key in kwarg: %s" % k)
                    if isinstance(v, bool):
                        setattr(_config, k, v)
                    else:
                        msg = "Unexpected value for key: %s, got:%s"
                        raise ValueError(msg % (k, v))
            elif isinstance(kwarg, set):
                for item in kwarg:
                    if item not in _config.__dict__:
                        raise ValueError("Unexpected key in kwarg: %s" % item)
                    else:
                        setattr(_config, item, True)
            else:
                msg = "Unhandled configuration type for kwarg %s"
                raise ValueError(msg % type(kwarg))

        parse_config(_highlight, highlight)
        parse_config(_interleave, interleave)

        # This is the colour scheme. The graphviz HTML label renderer only takes
        # names for colours: https://www.graphviz.org/doc/info/shapes.html#html
        cs = defaultdict(lambda: 'white') # default bg colour is white
        cs['marker'] = 'orange'
        cs['python'] = 'yellow'
        cs['truebr'] = 'green'
        cs['falsebr'] = 'red'
        cs['incref'] = 'cyan'
        cs['decref'] = 'turquoise'
        cs['raise'] = 'lightpink'
        cs['meminfo'] = 'lightseagreen'
        cs['return'] = 'purple'
        cs['llvm_intrin_calls'] = 'rosybrown'
        cs['function_calls'] = 'tomato'

        # Get the raw dot format information from LLVM and the LLVM IR
        fn = self.cres.get_function(self.name)
        #raw_dot = ll.get_function_cfg(fn).replace('\\l...', '')
        llvm_str = self.cres.get_llvm_str()

        def get_metadata(llvm_str):
            """ Gets the metadata entries from the LLVM IR, these look something
            like '!123 = INFORMATION'. Returns a map of metadata key to metadata
            value, i.e. from the example {'!123': INFORMATION}"""
            md = {}
            metadata_entry = re.compile(r'(^[!][0-9]+)(\s+=\s+.*)')
            for x in llvm_str.splitlines():
                match = metadata_entry.match(x)
                if match is not None:
                    g = match.groups()
                    if g is not None:
                        assert len(g) == 2
                        md[g[0]] = g[1]
            return md

        md = get_metadata(llvm_str)

        # setup digraph with initial properties
        def init_digraph(name, fname, fontsize):
            # name and fname are arbitrary graph and file names, they appear in
            # some rendering formats, the fontsize determines the output
            # fontsize.

            # truncate massive mangled names as file names as it causes OSError
            # when trying to render to pdf
            cmax = 200
            if len(fname) > cmax:
                wstr = (f'CFG output filename "{fname}" exceeds maximum '
                        f'supported length, it will be truncated.')
                warnings.warn(wstr, NumbaInvalidConfigWarning)
                fname = fname[:cmax]
            f = gv.Digraph(name, filename=fname)
            f.attr(rankdir='TB')
            f.attr('node', shape='none', fontsize='%s' % str(fontsize))
            return f

        f = init_digraph(self.name, self.name, fontsize)

        # A lot of regex is needed to parse the raw dot output. This output
        # contains a mix of LLVM IR in the labels, and also DOT markup.

        # DOT syntax, matches a "port" (where the tail of an edge starts)
        port_match = re.compile('.*{(.*)}.*')
        # DOT syntax, matches the "port" value from a found "port_match"
        port_jmp_match = re.compile('.*<(.*)>(.*)')
        # LLVM syntax, matches a LLVM debug marker
        metadata_marker = re.compile(r'.*!dbg\s+(![0-9]+).*')
        # LLVM syntax, matches a location entry
        location_expr = (r'.*!DILocation\(line:\s+([0-9]+),'
                         r'\s+column:\s+([0-9]),.*')
        location_entry = re.compile(location_expr)
        # LLVM syntax, matches LLVMs internal debug value calls
        dbg_value = re.compile(r'.*call void @llvm.dbg.value.*')
        # LLVM syntax, matches tokens for highlighting
        nrt_incref = re.compile(r"@NRT_incref\b")
        nrt_decref = re.compile(r"@NRT_decref\b")
        nrt_meminfo = re.compile("@NRT_MemInfo")
        ll_intrin_calls = re.compile(r".*call.*@llvm\..*")
        ll_function_call = re.compile(r".*call.*@.*")
        ll_raise = re.compile(r"store .*\!numba_exception_output.*")
        ll_return = re.compile("ret i32 [^1],?.*")

        # wrapper function for line wrapping LLVM lines
        def wrap(s):
            return textwrap.wrap(s, width=120, subsequent_indent='... ')

        # function to fix (sometimes escaped for DOT!) LLVM IR etc that needs to
        # be HTML escaped
        def clean(s):
            # Grab first 300 chars only, 1. this should be enough to identify
            # the token and it keeps names short. 2. graphviz/dot has a maximum
            # buffer size near 585?!, with additional transforms it's hard to
            # know if this would be exceeded. 3. hash of the token string is
            # written into the rendering to permit exact identification against
            # e.g. LLVM IR dump if necessary.
            n = 300
            if len(s) > n:
                hs = str(hash(s))
                s = '{}...<hash={}>'.format(s[:n], hs)
            s = html.escape(s) # deals with  &, < and >
            s = s.replace('\\{', "&#123;")
            s = s.replace('\\}', "&#125;")
            s = s.replace('\\', "&#92;")
            s = s.replace('%', "&#37;")
            s = s.replace('!', "&#33;")
            return s

        # These hold the node and edge ids from the raw dot information. They
        # are used later to wire up a new DiGraph that has the same structure
        # as the raw dot but with new nodes.
        node_ids = {}
        edge_ids = {}

        # Python source lines, used if python source interleave is requested
        if _interleave.python:
            src_code, firstlineno = inspect.getsourcelines(self.py_func)

        # This is the dot info from LLVM, it's in DOT form and has continuation
        # lines, strip them and then re-parse into `dot_json` form for use in
        # producing a formatted output.
        raw_dot = ll.get_function_cfg(fn).replace('\\l...', '')
        json_bytes = gv.Source(raw_dot).pipe(format='dot_json')
        jzon = json.loads(json_bytes.decode('utf-8'))

        idc = 0
        # Walk the "objects" (nodes) in the DOT output
        for obj in jzon['objects']:
            # These are used to keep tabs on the current line and column numbers
            # as per the markers. They are tracked so as to make sure a marker
            # is only emitted if there's a change in the marker.
            cur_line, cur_col = -1, -1
            label = obj['label']
            name = obj['name']
            gvid = obj['_gvid']
            node_ids[gvid] = name
            # Label is DOT format, it needs the head and tail removing and then
            # splitting for walking.
            label = label[1:-1]
            lines = label.split('\\l')

            # Holds the new lines
            new_lines = []

            # Aim is to produce an HTML table a bit like this:
            #
            # |------------|
            # | HEADER     | <-- this is the block header
            # |------------|
            # | LLVM SRC   | <--
            # | Marker?    | < this is the label/block body
            # | Python src?| <--
            # |------------|
            # | T   |  F   |  <-- this is the "ports", also determines col_span
            # --------------
            #

            # This is HTML syntax, its the column span. If there's a switch or a
            # branch at the bottom of the node this is rendered as multiple
            # columns in a table. First job is to go and render that and work
            # out how many columns are needed as that dictates how many columns
            # the rest of the source lines must span. In DOT syntax the places
            # that edges join nodes are referred to as "ports". Syntax in DOT
            # is like `node:port`.
            col_span = 1

            # First see if there is a port entry for this node
            port_line = ''
            matched = port_match.match(lines[-1])
            sliced_lines = lines
            if matched is not None:
                # There is a port
                ports = matched.groups()[0]
                ports_tokens = ports.split('|')
                col_span = len(ports_tokens)
                # Generate HTML table data cells, one for each port. If the
                # ports correspond to a branch then they can optionally
                # highlighted based on T/F.
                tdfmt = ('<td BGCOLOR="{}" BORDER="1" ALIGN="center" '
                         'PORT="{}">{}</td>')
                tbl_data = []
                if _highlight.branches:
                    colors = {'T': cs['truebr'], 'F': cs['falsebr']}
                else:
                    colors = {}
                for tok in ports_tokens:
                    target, value = port_jmp_match.match(tok).groups()
                    color = colors.get(value, 'white')
                    tbl_data.append(tdfmt.format(color, target, value))
                port_line = ''.join(tbl_data)
                # Drop the last line from the rest of the parse as it's the port
                # and just been dealt with.
                sliced_lines = lines[:-1]

            # loop peel the block header, it needs a HTML border
            fmtheader = ('<tr><td BGCOLOR="{}" BORDER="1" ALIGN="left" '
                         'COLSPAN="{}">{}</td></tr>')
            new_lines.append(fmtheader.format(cs['default'], col_span,
                                              clean(sliced_lines[0].strip())))

            # process rest of block creating the table row at a time.
            fmt = ('<tr><td BGCOLOR="{}" BORDER="0" ALIGN="left" '
                   'COLSPAN="{}">{}</td></tr>')

            def metadata_interleave(l, new_lines):
                """
                Search line `l` for metadata associated with python or line info
                and inject it into `new_lines` if requested.
                """
                matched = metadata_marker.match(l)
                if matched is not None:
                    # there's a metadata marker
                    g = matched.groups()
                    if g is not None:
                        assert len(g) == 1, g
                        marker = g[0]
                        debug_data = md.get(marker, None)
                        if debug_data is not None:
                            # and the metadata marker has a corresponding piece
                            # of metadata
                            ld = location_entry.match(debug_data)
                            if ld is not None:
                                # and the metadata is line info... proceed
                                assert len(ld.groups()) == 2, ld
                                line, col = ld.groups()
                                # only emit a new marker if the line number in
                                # the metadata is "new".
                                if line != cur_line or col != cur_col:
                                    if _interleave.lineinfo:
                                        mfmt = 'Marker %s, Line %s, column %s'
                                        mark_line = mfmt % (marker, line, col)
                                        ln = fmt.format(cs['marker'], col_span,
                                                        clean(mark_line))
                                        new_lines.append(ln)
                                    if _interleave.python:
                                        # TODO:
                                        # +1 for decorator, this probably needs
                                        # the same thing doing as for the
                                        # error messages where the decorator
                                        # is scanned for, its not always +1!
                                        lidx = int(line) - (firstlineno + 1)
                                        source_line = src_code[lidx + 1]
                                        ln = fmt.format(cs['python'], col_span,
                                                        clean(source_line))
                                        new_lines.append(ln)
                                    return line, col

            for l in sliced_lines[1:]:

                # Drop LLVM debug call entries
                if dbg_value.match(l):
                    continue

                # if requested generate interleaving of markers or python from
                # metadata
                if _interleave.lineinfo or _interleave.python:
                    updated_lineinfo = metadata_interleave(l, new_lines)
                    if updated_lineinfo is not None:
                        cur_line, cur_col = updated_lineinfo

                # Highlight other LLVM features if requested, HTML BGCOLOR
                # property is set by this.
                if _highlight.incref and nrt_incref.search(l):
                    colour = cs['incref']
                elif _highlight.decref and nrt_decref.search(l):
                    colour = cs['decref']
                elif _highlight.meminfo and nrt_meminfo.search(l):
                    colour = cs['meminfo']
                elif _highlight.raises and ll_raise.search(l):
                    # search for raise as its more specific than exit
                    colour = cs['raise']
                elif _highlight.returns and ll_return.search(l):
                    colour = cs['return']
                elif _highlight.llvm_intrin_calls and ll_intrin_calls.search(l):
                    colour = cs['llvm_intrin_calls']
                elif _highlight.function_calls and ll_function_call.search(l):
                    colour = cs['function_calls']
                else:
                    colour = cs['default']

                # Use the default coloring as a flag to force printing if a
                # special token print was requested AND LLVM ir stripping is
                # required
                if colour is not cs['default'] or not strip_ir:
                    for x in wrap(clean(l)):
                        new_lines.append(fmt.format(colour, col_span, x))

            # add in the port line at the end of the block if it was present
            # (this was built right at the top of the parse)
            if port_line:
                new_lines.append('<tr>{}</tr>'.format(port_line))

            # If there was data, create a table, else don't!
            dat = ''.join(new_lines)
            if dat:
                tab = (('<table id="%s" BORDER="1" CELLBORDER="0" '
                       'CELLPADDING="0" CELLSPACING="0">%s</table>') % (idc,
                                                                        dat))
                label = '<{}>'.format(tab)
            else:
                label = ''

            # finally, add a replacement node for the original with a new marked
            # up label.
            f.node(name, label=label)

        # Parse the edge data
        if 'edges' in jzon: # might be a single block, no edges
            for edge in jzon['edges']:
                gvid = edge['_gvid']
                tp = edge.get('tailport', None)
                edge_ids[gvid] = (edge['head'], edge['tail'], tp)

        # Write in the edge wiring with respect to the new nodes:ports.
        for gvid, edge in edge_ids.items():
            tail = node_ids[edge[1]]
            head = node_ids[edge[0]]
            port = edge[2]
            if port is not None:
                tail += ':%s' % port
            f.edge(tail, head)

        # Add a key to the graph if requested.
        if show_key:
            key_tab = []
            for k, v in cs.items():
                key_tab.append(('<tr><td BGCOLOR="{}" BORDER="0" ALIGN="center"'
                                '>{}</td></tr>').format(v, k))
            # The first < and last > are DOT syntax, rest is DOT HTML.
            f.node("Key", label=('<<table BORDER="1" CELLBORDER="1" '
                    'CELLPADDING="2" CELLSPACING="1"><tr><td BORDER="0">'
                    'Key:</td></tr>{}</table>>').format(''.join(key_tab)))

        # Render if required
        if filename is not None or view is not None:
            f.render(filename=filename, view=view, format=render_format)

        # Else pipe out a SVG
        return f.pipe(format='svg')

    def display(self, filename=None, format='pdf', view=False):
        """
        Plot the CFG.  In IPython notebook, the return image object can be
        inlined.

        The *filename* option can be set to a specific path for the rendered
        output to write to.  If *view* option is True, the plot is opened by
        the system default application for the image format (PDF). *format* can
        be any valid format string accepted by graphviz, default is 'pdf'.
        """
        rawbyt = self.pretty_printer(filename=filename, view=view,
                                     render_format=format, **self.kwargs)
        return rawbyt.decode('utf-8')

    def _repr_svg_(self):
        return self.pretty_printer(**self.kwargs).decode('utf-8')

    def __repr__(self):
        return self.dot


class CodeLibrary(metaclass=ABCMeta):
    """
    An interface for bundling LLVM code together and compiling it.
    It is tied to a *codegen* instance (e.g. JITCPUCodegen) that will
    determine how the LLVM code is transformed and linked together.
    """

    _finalized = False
    _object_caching_enabled = False
    _disable_inspection = False

    def __init__(self, codegen: "CPUCodegen", name: str):
        self._codegen = codegen
        self._name = name
        ptc_name = f"{self.__class__.__name__}({self._name!r})"
        self._recorded_timings = PassTimingsCollection(ptc_name)
        # Track names of the dynamic globals
        self._dynamic_globals = []

        self._reload_init = set()

    @property
    def has_dynamic_globals(self):
        self._ensure_finalized()
        return len(self._dynamic_globals) > 0

    @property
    def recorded_timings(self):
        return self._recorded_timings

    @property
    def codegen(self):
        """
        The codegen object owning this library.
        """
        return self._codegen

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "<Library %r at 0x%x>" % (self.name, id(self))

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError("operation impossible on finalized object %r"
                               % (self,))

    def _ensure_finalized(self):
        if not self._finalized:
            self.finalize()

    def create_ir_module(self, name):
        """
        Create an LLVM IR module for use by this library.
        """
        self._raise_if_finalized()
        ir_module = self._codegen._create_empty_module(name)
        return ir_module

    @abstractmethod
    def add_linking_library(self, library):
        """
        Add a library for linking into this library, without losing
        the original library.
        """

    @abstractmethod
    def add_ir_module(self, ir_module):
        """
        Add an LLVM IR module's contents to this library.
        """

    @abstractmethod
    def finalize(self):
        """
        Finalize the library.  After this call, nothing can be added anymore.
        Finalization involves various stages of code optimization and
        linking.
        """

    @abstractmethod
    def get_function(self, name):
        """
        Return the function named ``name``.
        """

    @abstractmethod
    def get_llvm_str(self):
        """
        Get the human-readable form of the LLVM module.
        """

    @abstractmethod
    def get_asm_str(self):
        """
        Get the human-readable assembly.
        """

    #
    # Object cache hooks and serialization
    #

    def enable_object_caching(self):
        self._object_caching_enabled = True
        self._compiled_object = None
        self._compiled = False

    def _get_compiled_object(self):
        if not self._object_caching_enabled:
            raise ValueError("object caching not enabled in %s" % (self,))
        if self._compiled_object is None:
            raise RuntimeError("no compiled object yet for %s" % (self,))
        return self._compiled_object

    def _set_compiled_object(self, value):
        if not self._object_caching_enabled:
            raise ValueError("object caching not enabled in %s" % (self,))
        if self._compiled:
            raise ValueError("library already compiled: %s" % (self,))
        self._compiled_object = value
        self._disable_inspection = True


class CPUCodeLibrary(CodeLibrary):

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self._linking_libraries = []   # maintain insertion order
        self._final_module = ll.parse_assembly(
            str(self._codegen._create_empty_module(self.name)))
        self._final_module.name = cgutils.normalize_ir_text(self.name)
        self._shared_module = None
        self._jit_handle = None

    def _optimize_functions(self, ll_module):
        """
        Internal: run function-level optimizations inside *ll_module*.
        """
        # Data layout deliberately *not* set here: the engine's compileAndAdd
        # path sets it from the engine's TargetMachine just before codegen, so
        # the MCJIT-era "set on every llvm.Module by hand" step is now redundant
        # and would only invite layout-string drift between Numba and the engine.
        for func in ll_module.functions:
            # Run function-level optimizations to reduce memory usage and improve
            # module-level optimization.
            fpm, pb = self._codegen._function_pass_manager()
            k = f"Function passes on {func.name!r}"
            with self._recorded_timings.record(k, pb):
                fpm.run(func, pb)

    def _optimize_final_module(self):

        """
        Internal: optimize this library's final module.
        """

        mpm_cheap, mpb_cheap =  self._codegen._module_pass_manager(
                                           loop_vectorize=self._codegen._loopvect,
                                           slp_vectorize=False,
                                           opt=self._codegen._opt_level,
                                           cost="cheap")

        mpm_full, mpb_full = self._codegen._module_pass_manager()
        cheap_name = "Module passes (cheap optimization for refprune)"
        with self._recorded_timings.record(cheap_name, mpb_cheap):
            # A cheaper optimisation pass is run first to try and get as many
            # refops into the same function as possible via inlining
            mpm_cheap.run(self._final_module, mpb_cheap)
        # Refop pruning is then run on the heavily inlined function
        if not config.LLVM_REFPRUNE_PASS:
            self._final_module = remove_redundant_nrt_refct(self._final_module)
        full_name = "Module passes (full optimization)"
        with self._recorded_timings.record(full_name, mpb_full):
            # The full optimisation suite is then run on the refop pruned IR
            mpm_full.run(self._final_module, mpb_full)

    def _get_module_for_linking(self):
        """
        Internal: get a LLVM module suitable for linking multiple times
        into another library.  Exported functions are made "linkonce_odr"
        to allow for multiple definitions, inlining, and removal of
        unused exports.

        See discussion in https://github.com/numba/numba/pull/890
        """
        self._ensure_finalized()
        if self._shared_module is not None:
            return self._shared_module
        mod = self._final_module
        to_fix = []
        nfuncs = 0
        for fn in mod.functions:
            nfuncs += 1
            if not fn.is_declaration and fn.linkage == ll.Linkage.external:
                to_fix.append(fn.name)
        if nfuncs == 0:
            # This is an issue which can occur if loading a module
            # from an object file and trying to link with it, so detect it
            # here to make debugging easier.
            raise RuntimeError("library unfit for linking: "
                               "no available functions in %s"
                               % (self,))
        if to_fix:
            mod = mod.clone()
            for name in to_fix:
                # NOTE: this will mark the symbol WEAK if serialized
                # to an ELF file
                mod.get_function(name).linkage = 'linkonce_odr'
        self._shared_module = mod
        return mod

    def add_linking_library(self, library):
        library._ensure_finalized()
        self._linking_libraries.append(library)

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module)
        ir = cgutils.normalize_ir_text(str(ir_module))
        ll_module = ll.parse_assembly(ir)
        ll_module.name = ir_module.name
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        self._optimize_functions(ll_module)
        # TODO: we shouldn't need to recreate the LLVM module object
        if not config.LLVM_REFPRUNE_PASS:
            ll_module = remove_redundant_nrt_refct(ll_module)
        self._final_module.link_in(ll_module)

    def finalize(self):
        require_global_compiler_lock()

        # Report any LLVM-related problems to the user
        self._codegen._check_llvm_bugs()

        self._raise_if_finalized()

        if config.DUMP_FUNC_OPT:
            dump("FUNCTION OPTIMIZED DUMP %s" % self.name,
                 self.get_llvm_str(), 'llvm')

        # Link libraries for shared code
        seen = set()
        for library in self._linking_libraries:
            if library not in seen:
                # Parent inherits reload_init
                self._reload_init.update(library._reload_init)
                seen.add(library)
                self._final_module.link_in(
                    library._get_module_for_linking(), preserve=True,
                )

        # Optimize the module after all dependences are linked in above,
        # to allow for inlining.
        self._optimize_final_module()

        self._final_module.verify()
        self._finalize_final_module()

    def _finalize_dynamic_globals(self):
        # Scan for dynamic globals
        for gv in self._final_module.global_variables:
            if gv.name.startswith('numba.dynamic.globals'):
                self._dynamic_globals.append(gv.name)

    def _verify_declare_only_symbols(self):
        # Verify that no declare-only function compiled by numba.
        for fn in self._final_module.functions:
            # We will only check for symbol name starting with '_ZN5numba'
            if fn.is_declaration and fn.name.startswith('_ZN5numba'):
                msg = 'Symbol {} not linked properly'
                raise AssertionError(msg.format(fn.name))

    def _finalize_final_module(self):
        """
        Make the underlying LLVM module ready to use.
        """
        self._finalize_dynamic_globals()
        self._verify_declare_only_symbols()

        # Register this library under the module identifier *before*
        # _add_module: the engine fires its notify callback synchronously
        # from inside compileAndAdd (within _add_module) once the framed
        # blob is ready, and the trampoline keys back into _libs_by_module_id
        # to find the owning library. Registering after _add_module would
        # miss that fire-and-forget call. Module identifier == library name
        # here; collisions are not currently a concern because two libraries
        # never share a name in the same Codegen.
        regs = getattr(self._codegen, '_libs_by_module_id', None)
        if regs is not None:
            regs[self._final_module.name] = weakref.ref(self)

        # _add_module() must be done after all linking_libraries have been
        # link_in'd so the inliner can act on the merged module; doing it
        # earlier breaks subsequent get_pointer_to_function() calls.
        # AOTCPUCodegen._add_module returns None — AOT has no JIT engine to
        # publish symbols into, only object emission.
        _orcjit_debug_dump_module_symbols(self._name, self._final_module)
        self._jit_handle = self._codegen._add_module(self._final_module)
        _orcjit_debug_dump_rename_map(self._codegen._engine,
                                      self._jit_handle, self._final_module)

        self._finalize_specific()

        self._finalized = True

        if config.DUMP_OPTIMIZED:
            dump("OPTIMIZED DUMP %s" % self.name, self.get_llvm_str(), 'llvm')

        if config.DUMP_ASSEMBLY:
            dump("ASSEMBLY %s" % self.name, self.get_asm_str(), 'asm')

    def get_defined_functions(self):
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
        mod = self._final_module
        for fn in mod.functions:
            if not fn.is_declaration:
                yield fn

    def get_function(self, name):
        return self._final_module.get_function(name)

    def _sentry_cache_disable_inspection(self):
        if self._disable_inspection:
            warnings.warn('Inspection disabled for cached code. '
                          'Invalid result is returned.')

    def get_llvm_str(self):
        self._sentry_cache_disable_inspection()
        return str(self._final_module)

    def get_asm_str(self):
        self._sentry_cache_disable_inspection()
        return str(self._codegen._tm.emit_assembly(self._final_module))

    def get_function_cfg(self, name, py_func=None, **kwargs):
        """
        Get control-flow graph of the LLVM function
        """
        self._sentry_cache_disable_inspection()
        return _CFG(self, name, py_func, **kwargs)

    def get_disasm_cfg(self, mangled_name):
        """
        Get the CFG of the disassembly of the ELF object at symbol mangled_name.

        Requires python package: r2pipe
        Requires radare2 binary on $PATH.
        Notebook rendering requires python package: graphviz
        Optionally requires a compiler toolchain (via pycc) to link the ELF to
        get better disassembly results.
        """
        elf = self._get_compiled_object()
        return disassemble_elf_to_cfg(elf, mangled_name)

    @classmethod
    def _dump_elf(cls, buf):
        """
        Dump the symbol table of an ELF file.
        Needs pyelftools (https://github.com/eliben/pyelftools)
        """
        from elftools.elf.elffile import ELFFile
        from elftools.elf import descriptions
        from io import BytesIO
        f = ELFFile(BytesIO(buf))
        print("ELF file:")
        for sec in f.iter_sections():
            if sec['sh_type'] == 'SHT_SYMTAB':
                symbols = sorted(sec.iter_symbols(), key=lambda sym: sym.name)
                print("    symbols:")
                for sym in symbols:
                    if not sym.name:
                        continue
                    print("    - %r: size=%d, value=0x%x, type=%s, bind=%s"
                          % (sym.name.decode(),
                             sym['st_size'],
                             sym['st_value'],
                             descriptions.describe_symbol_type(sym['st_info']['type']),
                             descriptions.describe_symbol_bind(sym['st_info']['bind']),
                             ))
        print()

    def _on_object_compiled(self, buf):
        """
        Engine notify hit for this library's module. `buf` is the framed
        object blob (NORC header + rename map + object bytes) ready for
        on-disk caching. Captured here so serialize_using_object_code()
        can return it later from `_get_compiled_object`.
        """
        if self._object_caching_enabled:
            self._compiled = True
            self._compiled_object = buf

    def _on_object_getbuffer(self):
        """
        Source-compat shim. Under Option 5 cache hits load through
        add_object_file directly (see `_unserialize` kind='object'); the
        engine never queries this on the hit path. Retained because
        CPUCodegen._init still wires a `get` callback; the engine
        accepts the wiring but does not call it.
        """
        if self._object_caching_enabled and self._compiled_object:
            buf = self._compiled_object
            self._compiled_object = None
            return buf
        return None

    def serialize_using_bitcode(self):
        """
        Serialize this library using its bitcode as the cached representation.
        """
        self._ensure_finalized()
        return (self.name, 'bitcode', self._final_module.as_bitcode())

    def serialize_using_object_code(self):
        """
        Serialize this library using its object code as the cached
        representation.  We also include its bitcode for further inlining
        with other libraries.
        """
        self._ensure_finalized()
        data = (self._get_compiled_object(),
                self._get_module_for_linking().as_bitcode())
        return (self.name, 'object', data)

    @classmethod
    def _unserialize(cls, codegen, state):
        name, kind, data = state
        self = codegen.create_library(name)
        assert isinstance(self, cls)
        if kind == 'bitcode':
            # No need to re-run optimizations, just make the module ready
            self._final_module = ll.parse_bitcode(data)
            self._finalize_final_module()
            return self
        elif kind == 'object':
            object_code, shared_bitcode = data
            # Keep the bytes for re-serialization (chained caching) and
            # disassembly (`get_disasm_cfg`).
            self.enable_object_caching()
            self._set_compiled_object(object_code)
            # `shared_bitcode` is only used as a source for downstream
            # `_get_module_for_linking` calls when *this* cached library is
            # itself link_in'd into another library. The engine sees only
            # the object bytes.
            self._shared_module = ll.parse_bitcode(shared_bitcode)
            # Cache-replay path: feed the framed object blob straight into
            # the engine. The framed prefix carries the rename map captured
            # at compile time; addObjectFile parses it and installs the map
            # on the returned handle, so per-handle lookups still translate
            # original → cached-renamed names without recomputing hashes.
            # Critically we do NOT round-trip through compileAndAdd here —
            # re-running the renamer on the bitcode could produce different
            # hashes (e.g. if the bitcode has been linked-in helpers it
            # would otherwise see fresh) and mismatch the .o's symtab.
            self._jit_handle = self._codegen._engine.add_object_file(
                object_code, name=self.name)
            # The cached .o has `.numba.unresolved$<sym>` external decls as
            # placeholders for recursive call sites. The live-compile path
            # patches them via add_global_mapping inside
            # _scan_and_fix_unresolved_refs; we must do the same on replay
            # against the shared bitcode, which still carries the decls.
            # Side effect: records this library's defined function names
            # against `_jit_handle` so a sibling library finalizing later
            # in the same process can resolve cross-module recursion
            # against the post-rename names from this library.
            self._codegen._scan_and_fix_unresolved_refs(self._shared_module,
                                                       self._jit_handle)
            self._finalized = True
            return self
        else:
            raise ValueError("unsupported serialization kind %r" % (kind,))


class AOTCodeLibrary(CPUCodeLibrary):

    def emit_native_object(self):
        """
        Return this library as a native object (a bytestring) -- for example
        ELF under Linux.

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._codegen._tm.emit_object(self._final_module)

    def emit_bitcode(self):
        """
        Return this library as LLVM bitcode (a bytestring).

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._final_module.as_bitcode()

    def _finalize_specific(self):
        pass


class JITCodeLibrary(CPUCodeLibrary):

    def get_pointer_to_function(self, name):
        """
        Generate native code for function named *name* and return a pointer
        to the start of the function (as an integer).

        This function implicitly calls .finalize().

        Returns
        -------
        pointer : int
            - zero (null) if no symbol of *name* is defined by this code
              library.
            - non-zero if the symbol is defined.
        """
        self._ensure_finalized()
        ee = self._codegen._engine
        defined = ee.is_symbol_defined(name, handle=self._jit_handle)
        if not defined:
            _orcjit_debug_lookup(self._name, self._jit_handle, name, False)
            return 0
        addr = ee.get_function_address(name, handle=self._jit_handle)
        _orcjit_debug_lookup(self._name, self._jit_handle, name, True, addr)
        return addr

    def _finalize_specific(self):
        self._codegen._scan_and_fix_unresolved_refs(self._final_module,
                                                    self._jit_handle)
        _orcjit_debug_finalize(self._name, self._jit_handle, "start")
        with self._recorded_timings.record_legacy("Finalize object"):
            self._codegen._engine.finalize_object()
        _orcjit_debug_finalize(self._name, self._jit_handle, "done")

    def set_env(self, env_name, env):
        """Patch this library's env GV named *env_name* to point at *env*.

        Used to be a Codegen-level helper that did a flat
        get_global_value_address. It must now be library-scoped: env GVs
        are content-hash-renamed in lockstep with their referencing
        function (via !numba.env_for, see base.declare_env_global), so the
        original `env_name` only resolves to storage if translated through
        *this* library's per-handle rename map.
        """
        gvaddr = self.get_pointer_to_function(env_name)
        envptr = (ctypes.c_void_p * 1).from_address(gvaddr)
        envptr[0] = ctypes.c_void_p(id(env))


class RuntimeLinker(object):
    """
    For tracking unresolved symbols generated at runtime due to recursion.
    """
    PREFIX = '.numba.unresolved$'

    def __init__(self):
        self._unresolved = utils.UniqueDict()
        # IR-level name -> defining library's JIT handle. Used to be a flat
        # set in MCJIT days when the engine had one symbol table. Under the
        # OrcJIT renamer, each library carries its own old→new map keyed
        # by handle; recording the handle at scan time lets `resolve()`
        # translate the IR name to its post-rename linker name in the
        # owning library's map.
        self._defined = {}
        self._resolved = []

    def scan_unresolved_symbols(self, module, engine):
        """
        Scan and track all unresolved external symbols in the module and
        allocate memory for it.
        """
        prefix = self.PREFIX

        for gv in module.global_variables:
            if gv.name.startswith(prefix):
                sym = gv.name[len(prefix):]
                # Avoid remapping to existing GV
                if engine.is_symbol_defined(gv.name):
                    continue
                # Allocate a memory space for the pointer
                abortfn = rtsys.library.get_pointer_to_function("nrt_unresolved_abort")
                ptr = ctypes.c_void_p(abortfn)
                engine.add_global_mapping(gv, ctypes.addressof(ptr))
                self._unresolved[sym] = ptr

    def scan_defined_symbols(self, module, handle):
        """
        Scan and track all defined symbols, recording the *handle* of the
        library that owns each one.
        """
        for fn in module.functions:
            if not fn.is_declaration:
                self._defined[fn.name] = handle

    def resolve(self, engine):
        """
        Fix unresolved symbols if they are defined.
        """
        # An iterator to get all unresolved but available symbols
        pending = [name for name in self._unresolved if name in self._defined]
        results = []
        # Resolve pending symbols
        for name in pending:
            # Get runtime address: handle-aware lookup, since the engine's
            # renamer rewrote the IR-level name to `name_<hash>`.
            fnptr = engine.get_function_address(name, handle=self._defined[name])
            results.append((name, fnptr))
            # Fix all usage
            ptr = self._unresolved[name]
            ptr.value = fnptr
            self._resolved.append((name, ptr))   # keep ptr alive
            # Delete resolved
            del self._unresolved[name]
        _orcjit_debug_resolve(pending, results)

def _proxy(old):
    @functools.wraps(old)
    def wrapper(self, *args, **kwargs):
        return old(self._ee, *args, **kwargs)
    return wrapper


class JitEngine(object):
    """Thin wrapper over NewOrcJIT preserving the MCJIT-era call shape.

    Pre-OrcJIT this class carried Numba-side bookkeeping for "which
    symbols does this engine know about" (a Python-side `_defined_symbols`
    set mirroring what MCJIT couldn't tell us). Under OrcJIT all that
    state lives in the binding: rename maps, defined-name sets per handle,
    process-symbol resolver chain. The methods below are a stable façade
    for `codegen.py` and `compiler.py` so we did not have to rewrite every
    `engine.<x>` call site to talk to NewOrcJIT directly.
    """
    def __init__(self, ee):
        self._ee = ee

    def is_symbol_defined(self, name, handle=None):
        return self._ee.is_symbol_defined(name, handle=handle)

    def add_module(self, module):
        """Add an llvmlite Module; return the engine's opaque handle."""
        return self._ee.add_ir_module(module)

    def add_object_file(self, blob, name=None):
        """Add precompiled object bytes (framed or raw); return a handle.

        Used by the cache-replay path to bypass the IR compile layer
        entirely — the engine consumes the object directly, no rename
        pass, no IRCompileLayer promise.
        """
        if name is None:
            return self._ee.add_object_file(blob)
        return self._ee.add_object_file(blob, name=name)

    def add_global_mapping(self, gv, addr):
        # Binding accepts either a string or an object with a .name attribute.
        return self._ee.add_global_mapping(gv, addr)

    def get_function_address(self, name, handle=None):
        return self._ee.get_function_address(name, handle=handle)

    def get_global_value_address(self, name: str):
        # No handle: cross-library lookup against the flat linker-name table.
        return self._ee.get_function_address(name)

    #
    # Re-exports of the underlying engine APIs.
    #
    set_object_cache = _proxy(ll.NewOrcJIT.set_object_cache)
    finalize_object = _proxy(ll.NewOrcJIT.finalize_object)


class Codegen(metaclass=ABCMeta):
    """
    Base Codegen class. It is expected that subclasses set the class attribute
    ``_library_class``, indicating the CodeLibrary class for the target.

    Subclasses should also initialize:

    ``self._data_layout``: the data layout for the target.
    ``self._target_data``: the binding layer ``TargetData`` for the target.
    """

    @abstractmethod
    def _create_empty_module(self, name):
        """
        Create a new empty module suitable for the target.
        """

    @abstractmethod
    def _add_module(self, module):
        """
        Add a module to the execution engine. Ownership of the module is
        transferred to the engine.
        """

    @property
    def target_data(self):
        """
        The LLVM "target data" object for this codegen instance.
        """
        return self._target_data

    def create_library(self, name, **kwargs):
        """
        Create a :class:`CodeLibrary` object for use with this codegen
        instance.
        """
        return self._library_class(self, name, **kwargs)

    def unserialize_library(self, serialized):
        return self._library_class._unserialize(self, serialized)


class CPUCodegen(Codegen):

    def __init__(self, module_name):
        initialize_llvm()

        self._data_layout = None
        self._llvm_module = ll.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._llvm_module.name = "global_codegen_module"
        self._rtlinker = RuntimeLinker()
        self._init(self._llvm_module)

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"

        # Module-identifier → library weakref. Populated by each library in
        # _finalize_final_module right before _add_module. compileAndAdd
        # fires the notify trampoline synchronously, so the entry must be
        # in place at that point. WeakRef so dead libraries drop out
        # naturally — captures here would otherwise keep them alive across
        # the whole codegen lifetime.
        self._libs_by_module_id = {}

        target = ll.Target.from_triple(ll.get_process_triple())
        tm_options = dict(opt=config.OPT)
        self._tm_features = self._customize_tm_features()
        self._customize_tm_options(tm_options)
        tm = target.create_target_machine(**tm_options)
        # Engine is created with its own TargetMachine built from
        # JITTargetMachineBuilder::detectHost. The `tm` constructed above
        # is kept on self._tm for codegen-time uses (emit_assembly,
        # disassembly, AOT object emit) but does NOT drive the JIT — the
        # engine's internal TM does. Two TMs, same host triple, no
        # observable divergence.
        engine = ll.create_new_orcjit()
        # The seeded `llvm_module` is empty (see assert above); adding it
        # is a leftover from the MCJIT bootstrap, where the engine
        # demanded an initial module. Kept for shape parity; harmless.
        engine.add_ir_module(llvm_module)

        # ENABLE_PROFILING / enable_jit_events: low-priority gap (see
        # orcjit_migration.md Status). ORC's GDB/perf event hooks exist
        # but are not yet wired through.
        # if config.ENABLE_PROFILING:
        #     engine.enable_jit_events()

        self._tm = tm
        self._engine = JitEngine(engine)
        self._target_data = engine.target_data
        self._data_layout = str(self._target_data)

        if config.OPT.is_opt_max:
            # If the OPT level is set to 'max' then the user is requesting that
            # compilation time is traded for potential performance gain. This
            # currently manifests as running the "cheap" pass at -O3
            # optimisation level with loop-vectorization enabled. There's no
            # guarantee that this will increase runtime performance, it may
            # detriment it, this is here to give the user an easily accessible
            # option to try.
            self._loopvect = True
            self._opt_level = 3
        else:
            # The default behaviour is to do an opt=0 pass to try and inline as
            # much as possible with the cheapest cost of doing so. This is so
            # that the ref-op pruner pass that runs after the cheap pass will
            # have the largest possible scope for working on pruning references.
            self._loopvect = False
            self._opt_level = 0

        # Notify trampoline: fired by compileAndAdd in C++ once the framed
        # blob is ready. Routes back to the owning library so
        # `serialize_using_object_code` can return the bytes on disk.
        def _notify(key, blob):
            ref = self._libs_by_module_id.get(key)
            lib = ref() if ref is not None else None
            if lib is not None:
                lib._on_object_compiled(blob)

        # _get is kept for API symmetry with the MCJIT-era ObjectCache
        # contract; the engine accepts the wiring but never calls it
        # (cache hits go through library._unserialize → add_object_file
        # directly). Removing it from this signature would just force
        # callers to know that detail.
        def _get(key):
            ref = self._libs_by_module_id.get(key)
            lib = ref() if ref is not None else None
            if lib is None:
                return None
            return lib._on_object_getbuffer()

        self._engine.set_object_cache(_notify, _get)

    def _create_empty_module(self, name):
        ir_module = llvmir.Module(cgutils.normalize_ir_text(name))
        ir_module.triple = ll.get_process_triple()
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        return ir_module

    def _module_pass_manager(self, **kwargs):
        cost = kwargs.pop("cost", None)
        pb = self._pass_builder(**kwargs)
        pm = pb.getModulePassManager()
        # If config.OPT==0 do not include these extra passes to help with
        # vectorization.
        if cost is not None and cost == "cheap" and config.OPT != 0:
            # This knocks loops into rotated form early to reduce the likelihood
            # of vectorization failing due to unknown PHI nodes.
            pm.add_loop_rotate_pass()
            # These passes are required to get SVML to vectorize tests
            pm.add_instruction_combine_pass()
            pm.add_jump_threading_pass()

        if config.LLVM_REFPRUNE_PASS:
           pm.add_refprune_pass(_parse_refprune_flags())
        return pm, pb

    def _function_pass_manager(self, **kwargs):
        pb = self._pass_builder(**kwargs)
        pm = pb.getFunctionPassManager()
        if config.LLVM_REFPRUNE_PASS:
            pm.add_refprune_pass(_parse_refprune_flags())
        return pm, pb

    def _pass_builder(self, **kwargs):
        opt_level = kwargs.pop('opt', config.OPT)
        loop_vectorize = kwargs.pop('loop_vectorize', config.LOOP_VECTORIZE)
        slp_vectorize = kwargs.pop('slp_vectorize', config.SLP_VECTORIZE)

        pb = create_pass_builder(self._tm, opt=opt_level,
                                 loop_vectorize=loop_vectorize,
                                 slp_vectorize=slp_vectorize,
                                 **kwargs)

        return pb

    def _check_llvm_bugs(self):
        """
        Guard against some well-known LLVM bug(s).
        """
        # Check the locale bug at https://github.com/numba/numba/issues/1569
        # Note we can't cache the result as locale settings can change
        # across a process's lifetime.  Also, for this same reason,
        # the check here is a mere heuristic (there may be a race condition
        # between now and actually compiling IR).
        ir = """
            define double @func()
            {
                ret double 1.23e+01
            }
            """
        mod = ll.parse_assembly(ir)
        ir_out = str(mod)
        if "12.3" in ir_out or "1.23" in ir_out:
            # Everything ok
            return
        if "1.0" in ir_out:
            loc = locale.getlocale()
            raise RuntimeError(
                "LLVM will produce incorrect floating-point code "
                "in the current locale %s.\nPlease read "
                "https://numba.readthedocs.io/en/stable/user/faq.html#llvm-locale-bug "
                "for more information."
                % (loc,))
        raise AssertionError("Unexpected IR:\n%s\n" % (ir_out,))

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        return (self._llvm_module.triple, self._get_host_cpu_name(),
                self._tm_features)

    def _scan_and_fix_unresolved_refs(self, module, handle):
        self._rtlinker.scan_unresolved_symbols(module, self._engine)
        self._rtlinker.scan_defined_symbols(module, handle)
        self._rtlinker.resolve(self._engine)

    def insert_unresolved_ref(self, builder, fnty, name):
        voidptr = llvmir.IntType(8).as_pointer()
        ptrname = self._rtlinker.PREFIX + name
        llvm_mod = builder.module
        try:
            fnptr = llvm_mod.get_global(ptrname)
        except KeyError:
            # Not defined?
            fnptr = llvmir.GlobalVariable(llvm_mod, voidptr, name=ptrname)
            fnptr.linkage = 'external'
        return builder.bitcast(builder.load(fnptr), fnty.as_pointer())

    def _get_host_cpu_name(self):
        return (ll.get_host_cpu_name()
                if config.CPU_NAME is None
                else config.CPU_NAME)

    def _get_host_cpu_features(self):
        if config.CPU_FEATURES is not None:
            return config.CPU_FEATURES
        return get_host_cpu_features()


class AOTCPUCodegen(CPUCodegen):
    """
    A codegen implementation suitable for Ahead-Of-Time compilation
    (e.g. generation of object files).
    """

    _library_class = AOTCodeLibrary

    def __init__(self, module_name, cpu_name=None):
        # By default, use generic cpu model for the arch
        self._cpu_name = cpu_name or ''
        CPUCodegen.__init__(self, module_name)

    def _customize_tm_options(self, options):
        cpu_name = self._cpu_name
        if cpu_name == 'host':
            cpu_name = self._get_host_cpu_name()
        options['cpu'] = cpu_name
        options['reloc'] = 'pic'
        options['codemodel'] = 'default'
        options['features'] = self._tm_features

    def _customize_tm_features(self):
        # ISA features are selected according to the requested CPU model
        # in _customize_tm_options()
        return ''

    def _add_module(self, module):
        pass


class JITCPUCodegen(CPUCodegen):
    """
    A codegen implementation suitable for Just-In-Time compilation.
    """

    _library_class = JITCodeLibrary

    def _customize_tm_options(self, options):
        # As long as we don't want to ship the code to another machine,
        # we can specialize for this CPU.
        options['cpu'] = self._get_host_cpu_name()
        # LLVM 7 change: # https://reviews.llvm.org/D47211#inline-425406
        # JIT needs static relocation on x86*
        # native target is already initialized from base class __init__
        arch = ll.Target.from_default_triple().name
        if arch.startswith('x86'): # one of x86 or x86_64
            reloc_model = 'static'
        elif arch.startswith('ppc'):
            reloc_model = 'pic'
        else:
            reloc_model = 'default'
        options['reloc'] = reloc_model
        options['codemodel'] = 'jitdefault'

        # Set feature attributes (such as ISA extensions)
        # This overrides default feature selection by CPU model above
        options['features'] = self._tm_features

        # Deal with optional argument to ll.Target.create_target_machine
        sig = utils.pysignature(ll.Target.create_target_machine)
        if 'jit' in sig.parameters:
            # Mark that this is making a JIT engine
            options['jit'] = True

    def _customize_tm_features(self):
        # For JIT target, we will use LLVM to get the feature map
        return self._get_host_cpu_features()

    def _add_module(self, module):
        return self._engine.add_module(module)



def initialize_llvm():
    """Safe to use multiple times.
    """
    ll.initialize_native_target()
    ll.initialize_native_asmprinter()


def get_host_cpu_features():
    """Get host CPU features using LLVM.

    The features may be modified due to user setting.
    See numba.config.ENABLE_AVX.
    """
    try:
        features = ll.get_host_cpu_features()
    except RuntimeError:
        return ''
    else:
        if not config.ENABLE_AVX:
            # Disable all features with name starting with 'avx'
            for k in features:
                if k.startswith('avx'):
                    features[k] = False

        # Set feature attributes
        return features.flatten()
