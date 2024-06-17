import yaml
import re
import time
import os
import io
import platform
import html
from collections import defaultdict
import functools
from multiprocessing import Lock
import os, os.path
import subprocess
from sys import intern

import os.path
import shutil
import json
import glob
import pathlib
import collections
from pygments import highlight
from pygments.lexers.c_cpp import CppLexer
from pygments.formatters import HtmlFormatter
from itanium_demangler import parse as demangle

from numba.core import config

import logging

try:
    # Try to use the C parser.
    from yaml import CLoader as Loader
except ImportError:
    logging.warning("For faster parsing, you may want to install libYAML for PyYAML")
    from yaml import Loader


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def html_file_name(filename):
    replace_targets = ['/', '#', ':', '\\']
    new_name = filename
    for target in replace_targets:
        new_name = new_name.replace(target, '_')
    return new_name + ".html"


def make_link(File, Line):
    return "\"{}#L{}\"".format(html_file_name(File), Line)

class EmptyLock(object):
    def __enter__(self):
        return True
    def __exit__(self, *args):
        pass

class Remark(yaml.YAMLObject):
    # Work-around for http://pyyaml.org/ticket/154.
    yaml_loader = Loader

    default_demangler = 'c++filt -n -p'
    demangler_proc = None

#    @classmethod
#    def find_demangler(cls):


    @classmethod
    def open_demangler_proc(cls, demangler):
        cls.demangler_proc = subprocess.Popen(demangler.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    @classmethod
    def set_demangler(cls, demangler):
        cls.open_demangler_proc(demangler)
        if (platform.system() == 'Windows'):
            cls.demangler_lock = EmptyLock(); # on windows we spawn demangler for each process, no Lock is needed
        else:
            cls.demangler_lock = Lock()


    @classmethod
    def demangle(cls, name):
        if not cls.demangler_proc:
            cls.set_demangler(cls.default_demangler)
        with cls.demangler_lock:
            cls.demangler_proc.stdin.write((name + '\n').encode('utf-8'))
            cls.demangler_proc.stdin.flush()
            return cls.demangler_proc.stdout.readline().rstrip().decode('utf-8')

    # Intern all strings since we have lot of duplication across filenames,
    # remark text.
    #
    # Change Args from a list of dicts to a tuple of tuples.  This saves
    # memory in two ways.  One, a small tuple is significantly smaller than a
    # small dict.  Two, using tuple instead of list allows Args to be directly
    # used as part of the key (in Python only immutable types are hashable).
    def _reduce_memory(self):
        self.Pass = intern(self.Pass)
        self.Name = intern(self.Name)
        try:
            # Can't intern unicode strings.
            self.Function = intern(self.Function)
        except:
            pass

        def _reduce_memory_dict(old_dict):
            new_dict = dict()
            for (k, v) in old_dict.items():
                if type(k) is str:
                    k = intern(k)

                if type(v) is str:
                    v = intern(v)
                elif type(v) is dict:
                    # This handles [{'Caller': ..., 'DebugLoc': { 'File': ... }}]
                    v = _reduce_memory_dict(v)
                new_dict[k] = v
            return tuple(new_dict.items())

        self.Args = tuple([_reduce_memory_dict(arg_dict) for arg_dict in self.Args])

    # The inverse operation of the dictonary-related memory optimization in
    # _reduce_memory_dict.  E.g.
    #     (('DebugLoc', (('File', ...) ... ))) -> [{'DebugLoc': {'File': ...} ....}]
    def recover_yaml_structure(self):
        def tuple_to_dict(t):
            d = dict()
            for (k, v) in t:
                if type(v) is tuple:
                    v = tuple_to_dict(v)
                d[k] = v
            return d

        self.Args = [tuple_to_dict(arg_tuple) for arg_tuple in self.Args]

    def canonicalize(self):
        if not hasattr(self, 'Hotness'):
            self.Hotness = 0
        if not hasattr(self, 'Args'):
            self.Args = []
        self._reduce_memory()

    @property
    def File(self):
        return self.DebugLoc['File']

    @property
    def Line(self):
        return int(self.DebugLoc['Line'])

    @property
    def Column(self):
        return self.DebugLoc['Column']

    @property
    def DebugLocString(self):
        return "{}:{}:{}".format(self.File, self.Line, self.Column)

    @property
    def DemangledFunctionName(self):
        return self.demangle(self.Function)

    @property
    def Link(self):
        return make_link(self.File, self.Line)

    def getArgString(self, mapping):
        mapping = dict(list(mapping))
        dl = mapping.get('DebugLoc')
        if dl:
            del mapping['DebugLoc']

        assert(len(mapping) == 1)
        (key, value) = list(mapping.items())[0]

        if key == 'Caller' or key == 'Callee' or key == 'DirectCallee':
            value = html.escape(self.demangle(value))

        if dl and key != 'Caller':
            dl_dict = dict(list(dl))
            return u"<a href={}>{}</a>".format(
                make_link(dl_dict['File'], dl_dict['Line']), value)
        else:
            return value

    # Return a cached dictionary for the arguments.  The key for each entry is
    # the argument key (e.g. 'Callee' for inlining remarks.  The value is a
    # list containing the value (e.g. for 'Callee' the function) and
    # optionally a DebugLoc.
    def getArgDict(self):
        if hasattr(self, 'ArgDict'):
            return self.ArgDict
        self.ArgDict = {}
        for arg in self.Args:
            if len(arg) == 2:
                if arg[0][0] == 'DebugLoc':
                    dbgidx = 0
                else:
                    assert(arg[1][0] == 'DebugLoc')
                    dbgidx = 1

                key = arg[1 - dbgidx][0]
                entry = (arg[1 - dbgidx][1], arg[dbgidx][1])
            else:
                arg = arg[0]
                key = arg[0]
                entry = (arg[1], )

            self.ArgDict[key] = entry
        return self.ArgDict

    def getDiffPrefix(self):
        if hasattr(self, 'Added'):
            if self.Added:
                return '+'
            else:
                return '-'
        return ''

    @property
    def PassWithDiffPrefix(self):
        return self.getDiffPrefix() + self.Pass

    @property
    def message(self):
        # Args is a list of mappings (dictionaries)
        values = [self.getArgString(mapping) for mapping in self.Args]
        return "".join(values)

    @property
    def RelativeHotness(self):
        if self.max_hotness:
            return "{0:.2f}%".format(self.Hotness * 100. / self.max_hotness)
        else:
            return ''

    @property
    def key(self):
        return (self.__class__, self.PassWithDiffPrefix, self.Name, self.File,
                self.Line, self.Column, self.Function, self.Args)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __repr__(self):
        return str(self.key)


class Analysis(Remark):
    yaml_tag = '!Analysis'

    @property
    def color(self):
        return "white"


class AnalysisFPCommute(Analysis):
    yaml_tag = '!AnalysisFPCommute'


class AnalysisAliasing(Analysis):
    yaml_tag = '!AnalysisAliasing'


class Passed(Remark):
    yaml_tag = '!Passed'

    @property
    def color(self):
        return "green"


class Missed(Remark):
    yaml_tag = '!Missed'

    @property
    def color(self):
        return "red"

class Failure(Missed):
    yaml_tag = '!Failure'


# This allows passing the global context to the child processes.
class Context:
    def __init__(self, caller_loc = dict()):
       # Map function names to their source location for function where inlining happened
       self.caller_loc = caller_loc


def render_file_source(context, source_dir, output_dir, filename, line_remarks):
    html_filename = os.path.join(output_dir, html_file_name(filename))
    filename = filename if os.path.exists(filename) else os.path.join(source_dir, filename)

    html_formatter = HtmlFormatter(encoding='utf-8')
    cpp_lexer = CppLexer(stripnl=False)

    def render_source_lines(stream, line_remarks):
        file_text = stream.read()

        html_highlighted = highlight(
            file_text,
            cpp_lexer,
            html_formatter)

        # Note that the API is different between Python 2 and 3.  On
        # Python 3, pygments.highlight() returns a bytes object, so we
        # have to decode.  On Python 2, the output is str but since we
        # support unicode characters and the output streams is unicode we
        # decode too.
        html_highlighted = html_highlighted.decode('utf-8')

        # Take off the header and footer, these must be
        #   reapplied line-wise, within the page structure
        html_highlighted = html_highlighted.replace('<div class="highlight"><pre>', '')
        html_highlighted = html_highlighted.replace('</pre></div>', '')

        for (linenum, html_line) in enumerate(html_highlighted.split('\n'), start=1):
            yield [f'<a name="L{linenum}">{linenum}</a>', '', '', f'<div class="highlight"><pre>{html_line}</pre></div>', '']

            cur_line_remarks = line_remarks.get(linenum, [])
            from collections import defaultdict
            d = defaultdict(list)
            count_deleted = defaultdict(int)
            for obj in cur_line_remarks:
                if len(d[obj.Name]) < 5:
                    d[obj.Name].append(obj)
                else:
                    count_deleted[obj.Name] += 1

            for obj_name, remarks in d.items():
                # render caret line, if all rendered remarks share a column
                columns = [r.Column for r in remarks]
                if all(c == columns[0] for c in columns) and columns[0] != 0:
                    yield ['',
                       0,
                       {'class': f"column-entry-yellow", 'text': ''},
                       {'class': 'column-entry-yellow',
                        'text': f'''<span " class="indent-span">{"&nbsp"*(columns[0]-1) + '^'}&nbsp;</span>'''},
                       {'class': f"column-entry-yellow", 'text': ''},
                       ]
                for remark in remarks:
                    yield render_inline_remark(remark, html_line)
                if count_deleted[obj_name] != 0:
                    yield ['',
                        0,
                        {'class': f"column-entry-yellow", 'text': ''},
                        {'class': 'column-entry-yellow', 'text': f'''<span " class="indent-span">...{count_deleted[obj_name]} similar remarks omitted.&nbsp;</span>'''},
                        {'class': f"column-entry-yellow", 'text': ''},
                        ]


    def render_inline_remark(remark, line):
        inlining_context = remark.DemangledFunctionName
        dl = context.caller_loc.get(remark.Function)
        if dl:
            dl_dict = dict(list(dl))
            link = make_link(dl_dict['File'], dl_dict['Line'] - 2)
            inlining_context = f"<a href={link}>{remark.DemangledFunctionName}</a>"

        start_line = re.sub("^<span>", "", line)
        spaces = len(start_line) - len(start_line.lstrip())
        indent = f"{spaces + 2}ch"

        # Create expanded message and link if we have a multiline message.
        lines = remark.message.split('\n')
        if len(lines) > 1:
            expand_link = '<a style="text-decoration: none;" href="" onclick="toggleExpandedMessage(this); return false;">+</a>'
            message = lines[0]
            other_lines = "\n".join(lines[1:])
            expand_message = f'''
<div class="full-info" style="display:none;">
  <div class="expanded col-left" style="margin-left: {indent}"><pre>{other_lines}</pre></div>
</div>'''
        else:
            expand_link = ''
            expand_message = ''
            message = remark.message
        return ['',
                remark.RelativeHotness,
                {'class': f"column-entry-{remark.color}", 'text': remark.PassWithDiffPrefix},
                {'class': 'column-entry-yellow', 'text': f'''<span style="margin-left: {indent};" class="indent-span">&bull; {expand_link} {message}&nbsp;</span>{expand_message}'''},
                {'class': f"column-entry-yellow", 'text': inlining_context},
                ]

    with open(html_filename, "w", encoding='utf-8') as f: 
        if not os.path.exists(filename):
            f.write(f'''
    <html>
    <h1>Unable to locate file {filename}</h1>
</html>''')
            return

        try:
            with open(filename, encoding="utf8", errors='ignore') as source_stream:
                entries = list(render_source_lines(source_stream, line_remarks))
        except Exception:
            print(f"Failed to process file {filename}")
            raise

        f_string = f'''
<html>
<meta charset="utf-8" />
<head>
<title>{os.path.basename(filename)}</title>
<link rel="icon" type="image/png" href="assets/favicon.ico"/>
<link rel='stylesheet' type='text/css' href='assets/style.css'>
<link rel='stylesheet' type='text/css' href='assets/jquery.dataTables.min.css'>
<script src="assets/jquery-3.5.1.js"></script>
<script src="assets/jquery.dataTables.min.js"></script>
</head>
<body>
<h1 class="filename-title">{os.path.abspath(filename)}</h1>
<p><a class='back' href='index.html'>Back</a></p>
<table id="opt_table_code" class="" width="100%"></table>
<p><a class='back' href='index.html'>Back</a></p>

<script type="text/javascript">
var dataSet = {json.dumps(entries)};

function toggleExpandedMessage(e) {{
  var FullTextElems = e.parentElement.parentElement.getElementsByClassName("full-info");
  if (!FullTextElems || FullTextElems.length < 1) {{
      return false;
  }}
  var FullText = FullTextElems[0];
  if (FullText.style.display == 'none') {{
    e.innerHTML = '-';
    FullText.style.display = 'block';
  }} else {{
    e.innerHTML = '+';
    FullText.style.display = 'none';
  }}
}}

$(document).ready(function() {{
    $('#opt_table_code').DataTable( {{
        data: dataSet,
        paging: false,
        "ordering": false,
        "asStripeClasses": [],
        columns: [
            {{ title: "Line" }},
            {{ title: "Hotness" }},
            {{ title: "Optimization" }},
            {{ title: "Source" }},
            {{ title: "Inline Context" }}
        ],
        columnDefs: [
            {{
                "targets": "_all",
                "createdCell": function (td, data, rowData, row, col) {{
                    if (data.constructor == Object && data['class'] !== undefined) {{
                        $(td).addClass(data['class']);
                    }}
                }},
                "render": function(data, type, row) {{
                    if (data.constructor == Object && data['text'] !== undefined) {{
                        return data['text'];
                    }}
                    return data;
                }}
            }}
        ]
    }} );
    if (location.hash.length > 2) {{
        var loc = location.hash.split("#")[1];
        var aTag = $("a[name='" + loc + "']");
        if (aTag.length > 0) {{
            $('body').scrollTop(parseInt(aTag.offset().top));
        }}
    }}
}} );
</script>
</body>
</html>
'''
        f.write(f_string)
        f.close()
    
    print(f"Rendered {html_filename}")

def render_index(output_dir, all_remarks):
    def render_entry(remark):
        return dict(description=remark.Name,
                    loc=f"<a href={remark.Link}>{remark.DebugLocString}</a>",
                    message=remark.message,
                    functionName=remark.DemangledFunctionName,
                    relativeHotness=remark.RelativeHotness,
                    color=remark.color)

    entries = [render_entry(remark) for remark in all_remarks]

    entries_summary = collections.Counter(e['description'] for e in entries)
    entries_summary_li = '\n'.join(f"<li>{key}: {value}" for key, value in entries_summary.items())

    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(f'''
<html>
<meta charset="utf-8" />
<head>
<link rel="icon" type="image/png" href="assets/favicon.ico"/>
<link rel='stylesheet' type='text/css' href='assets/style.css'>
<link rel='stylesheet' type='text/css' href='assets/jquery.dataTables.min.css'>
<script src="assets/jquery-3.5.1.js"></script>
<script src="assets/jquery.dataTables.min.js"></script>
<script src="assets/colResizable-1.6.min.js"></script>
<title>OptView2 Index</title>
</head>
<body>
<h3>{len(entries_summary)} issue types:</h3>
<ul id='entries_summary'>
{entries_summary_li}
</ul>
<div class="centered">
<table id="opt_table" class="" width="100%"></table>
</div>
<script type="text/javascript">
var dataSet = {json.dumps(entries)};
$(document).ready(function() {{
    $('#opt_table').DataTable( {{
        data: dataSet,
        "lengthMenu": [[100, 500, -1], [100, 500, "All"]],
        columns: [
            {{ title: "Location", data: "loc" }},
            {{ title: "Description", data: "description" }},
            {{ title: "Function", data: "functionName" }},
            {{ title: "Message", data: "message" }},
            {{ title: "Hotness", data: "relativeHotness" }},
        ],
        columnDefs: [
            {{
                "targets": [1],
                "createdCell": function (td, data, rowData, row, col) {{
                    $(td).addClass("column-entry-" + rowData['color']);
                }},
            }}
        ]
    }} );
    $("#opt_table").colResizable()
}} );
</script>
</body>
</html>
''')
    return index_path



class RemarksNode:
    def __init__(self, function_name) -> None:
        self.function_name = function_name
        self.callers = []
        self.callees = []
        self.remarks = []

    def __repr__(self) -> str:
        return self.function_name


class RemarksInterface:
    def __enter__(self):
        self.original_remark_flag = config.LLVM_REMARKS
        self.original_remark_path = config.LLVM_REMARKS_FILE

        config.LLVM_REMARKS = 1
        self.file_path = self._get_opt_remarks_path()
        config.LLVM_REMARKS_FILE = self.file_path
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        config.LLVM_REMARKS = self.original_remark_flag
        config.LLVM_REMARKS_FILE = self.original_remark_path
        self.build_callgraph()

    def build_callgraph(self):
        # TODO: Do this in a more efficient way,
        # Reading everything at once may take up lots of memory
        # in case of larger functions.
        with open(self.file_path, "r") as f:
            remarks = f.read().split("---")

        self.callgraph = {}
        for remark in remarks:
            if not remark:
                continue
            # Load the YAML content
            # Get the first line of the remark
            status, remark = remark.split("\n", 1)
            remark_dict = yaml.safe_load(remark)
            remark_dict['Status'] = status.split("!")[-1]

            # Update function name by demangling and processing abi
            remark_dict["Function"] = self.process_raw_funcname(
                remark_dict["Function"]
            )

            remark_dict["Caller"] = None
            remark_dict["Callee"] = None

            # TODO: Process args properly
            # Make a single string out of the args
            # and save every other important info
            # as key-value within remark_dict
            for _x in remark_dict["Args"]:
                if "Caller" in _x:
                    # Update function name by demangling and processing abi
                    remark_dict["Caller"] = _x["Caller"] = \
                        self.process_raw_funcname(_x["Caller"])
                    caller_node = self.get_node(remark_dict["Caller"],
                                                create_new=True)

                if "Callee" in _x:
                    # Update function name by demangling and processing abi
                    remark_dict["Callee"] = _x["Callee"] = \
                        self.process_raw_funcname(_x["Callee"])
                    callee_node = self.get_node(remark_dict["Callee"],
                                                create_new=True)

            # Make graph connections
            if remark_dict["Callee"] and remark_dict["Caller"]:
                caller_node.callees.append(callee_node)
                callee_node.callers.append(caller_node)

            # Add the remark to the Function node
            function_node = self.get_node(remark_dict["Function"],
                                          create_new=True)
            function_node.remarks.append(remark_dict)

        self.head_nodes = [x for x in self.callgraph.values() if not x.callers]

    def get_node(self, func_name, create_new=False):
        try:
            return self.callgraph[func_name]
        except KeyError as e:
            if create_new:
                node = RemarksNode(func_name)
                self.callgraph[func_name] = node
                return node
            else:
                raise e

    @classmethod
    def process_raw_funcname(cls, func_name):
        func_name = cls.demangle_function_name(func_name)
        func_name = cls.process_abi(func_name)
        return func_name

    @classmethod
    def demangle_function_name(cls, func_name):
        if func_name.startswith("cfunc."):
            # Remove the cfunc prefix
            func_name = func_name[6:]

        demangled_name = demangle(func_name)
        if demangled_name is None:
            demangled_name = func_name
        else:
            demangled_name = str(demangled_name)
        return demangled_name

    @classmethod
    def process_abi(cls, function_name):
        if "[abi:" in function_name:
            # Get abi version and decode the flags
            func_split = re.split(r'[\[\]]', function_name)
            # Ignore abi for now
            return ''.join([_x for _x in func_split if
                            not _x.startswith("abi")])
        else:
            return function_name

    @classmethod
    def _get_opt_remarks_path(self):
        dirpath = 'numba_opt_remarks'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        # Avoid duplication of filename by using the id of this CodeLibrary
        # and the current timestamp.
        filename = '{:08x}-{}.opt.yaml'.format(id(self), time.time())
        remarks_file = os.path.join(dirpath, filename)

        return remarks_file

    def _gather_results(self, filename, exclude_names='', exclude_text='', collect_opt_success=False, annotate_external=False):
        logging.info('Reading YAML file...')

        max_hotness = 0
        all_remarks = dict()
        file_remarks = defaultdict(functools.partial(defaultdict, list))
        #logging.debug(f"Parsing {input_file}")

        #TODO: filter unique name+file+line loc *here*
        with io.open(filename, encoding = 'utf-8') as f:
            docs = yaml.load_all(f, Loader=Loader)

            exclude_text_e = None
            if exclude_text:
                exclude_text_e = re.compile(exclude_text)
            exclude_names_e = None
            if exclude_names:
                exclude_names_e = re.compile(exclude_names)
            for remark in docs:
                remark.canonicalize()
                # Avoid remarks withoug debug location or if they are duplicated
                if not hasattr(remark, 'DebugLoc') or remark.key in all_remarks:
                    continue

                if not collect_opt_success and not (isinstance(remark, Missed) | isinstance(remark, Failure)):
                    continue

                if not annotate_external:
                    if os.path.isabs(remark.File):
                        continue

                if exclude_names_e and exclude_names_e.search(remark.Name):
                    continue

                if exclude_text_e and exclude_text_e.search(remark.message):
                    continue

                all_remarks[remark.key] = remark

                file_remarks[remark.File][remark.Line].append(remark)

                # If we're reading a back a diff yaml file, max_hotness is already
                # captured which may actually be less than the max hotness found
                # in the file.
                if hasattr(remark, 'max_hotness'):
                    max_hotness = remark.max_hotness
                max_hotness = max(max_hotness, remark.Hotness)

        for filename, d in file_remarks.items():
            for line, remarks in d.items():
                for remark in remarks:
                    # Bring max_hotness into the remarks so that
                    # RelativeHotness does not depend on an external global.
                    remark.max_hotness = max_hotness

        return all_remarks, file_remarks, max_hotness != 0

    def generate_remarks(self, exclude_names='', exclude_text='', collect_opt_success=False, annotate_external=False):
        all_remarks, file_remarks, should_display_hotness = \
            self._gather_results(filename=self.file_path,
                                        exclude_names=exclude_names,
                                        exclude_text=exclude_text,
                                        collect_opt_success=collect_opt_success,
                                        annotate_external=annotate_external)

        self.context = Context()

        for remark in all_remarks.values():
            if isinstance(remark, Passed) and remark.Pass == "inline" and remark.Name == "Inlined":
                for arg in remark.Args:
                    arg_dict = dict(list(arg))
                    caller = arg_dict.get('Caller')
                    if caller:
                        try:
                            self.context.caller_loc[caller] = arg_dict['DebugLoc']
                        except KeyError:
                                pass
        
        self.all_remarks = all_remarks
        self.file_remarks = file_remarks
        self.should_display_hotness = should_display_hotness

    def render_remarks(self, output_dir, source_dir, open_browser=False, num_jobs=1):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        logging.info('Rendering index page...')
        logging.info(f"  {len(self.all_remarks):d} raw remarks")
        if len(self.all_remarks) == 0:
            logging.warning("Not generating report! Please verify your --source-dir argument is exactly the path from which the compiler was invoked.")
            return
            
        sorted_remarks = sorted(self.all_remarks.values(), key=lambda r: (r.File, r.Line, r.Column, r.PassWithDiffPrefix))
        unique_lines_remarks = [sorted_remarks[0]]
        for rmk in sorted_remarks:
            last_unq_rmk = unique_lines_remarks[-1]
            last_rmk_key = (last_unq_rmk.File, last_unq_rmk.Line, last_unq_rmk.Column, last_unq_rmk.PassWithDiffPrefix)
            rmk_key = (rmk.File, rmk.Line, rmk.Column, rmk.PassWithDiffPrefix)
            if rmk_key != last_rmk_key:
                unique_lines_remarks.append(rmk)
        logging.info("  {:d} unique source locations".format(len(unique_lines_remarks)))

        if self.should_display_hotness:
            sorted_remarks = sorted(unique_lines_remarks, key=lambda r: (r.Hotness, r.File, r.Line, r.Column, r.PassWithDiffPrefix, r.yaml_tag, r.Function), reverse=True)
        else:
            sorted_remarks = sorted(unique_lines_remarks, key=lambda r: (r.File, r.Line, r.Column, r.PassWithDiffPrefix, r.yaml_tag, r.Function))

        index_path = render_index(output_dir, sorted_remarks)

        # TODO: Fix asset copying
        logging.info("Copying assets")
        assets_path = pathlib.Path(output_dir) / "assets"
        assets_path.mkdir(parents=True, exist_ok=True)
        for filename in glob.glob(os.path.join(str(pathlib.Path(os.path.realpath(__file__)).parent), "assets", '*.*')):
            shutil.copy(filename, assets_path)


        logging.info('Rendering HTML files...')
        for filename, entry in self.file_remarks.items():
            render_file_source(self.context, source_dir, output_dir, filename, entry)
    
        url_path = f'file://{os.path.abspath(index_path)}'
        logging.info(f'Done - check the index page at {url_path}')
        if open_browser:
            try:
                import webbrowser
                if webbrowser.get("wslview %s") == None:
                    webbrowser.open(url_path)
                else:
                    webbrowser.get("wslview %s").open(url_path)
            except Exception:
                pass
