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

from numba.core import config

import logging

try:
    # Try to use the C parser.
    from yaml import CLoader as Loader
except ImportError:
    logging.warning("For faster parsing, you may "
                    "want to install libYAML for PyYAML")
    from yaml import Loader


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# flake8: noqa

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
    yaml_loader = Loader

    default_demangler = 'c++filt -n -p'
    demangler_proc = None

    @classmethod
    def open_demangler_proc(cls, demangler):
        cls.demangler_proc = subprocess.Popen(
            demangler.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    @classmethod
    def set_demangler(cls, demangler):
        cls.open_demangler_proc(demangler)
        if (platform.system() == 'Windows'):
            # on windows we spawn demangler for each process, no Lock is needed
            cls.demangler_lock = EmptyLock()
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
                    # This handles [{'Caller':...,'DebugLoc': { 'File': ... }}]
                    v = _reduce_memory_dict(v)
                new_dict[k] = v
            return tuple(new_dict.items())

        self.Args = tuple(
            [_reduce_memory_dict(arg_dict) for arg_dict in self.Args])

    # The inverse operation of the dictonary-related memory optimization in
    # _reduce_memory_dict.  E.g.
    # (('DebugLoc', (('File', ...) ... ))) -> [{'DebugLoc': {'File': ...} ....}]
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
    def __init__(self, caller_loc=dict()):
        # Map function names to their source location
        # for function where inlining happened
        self.caller_loc = caller_loc


class RemarksNode:
    def __init__(self, function_name) -> None:
        self.function_name = function_name
        self.callers = []
        self.callees = []
        self.remarks = []

    def __repr__(self) -> str:
        return self.function_name


class RemarksInterface:
    file_path = None

    def __enter__(self):
        self.original_remark_flag = config.LLVM_REMARKS
        self.original_remark_path = config.LLVM_REMARKS_FILE

        config.LLVM_REMARKS = 1
        if self.file_path is None:
            self.file_path = self._get_opt_remarks_path()
        config.LLVM_REMARKS_FILE = self.file_path
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        config.LLVM_REMARKS = self.original_remark_flag
        config.LLVM_REMARKS_FILE = self.original_remark_path

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

    def _gather_results(self, filename, exclude_names='', exclude_text='',
                        collect_passed_remarks=False, annotate_external=False):
        logging.info('Reading YAML file...')

        max_hotness = 0
        all_remarks = dict()
        file_remarks = defaultdict(functools.partial(defaultdict, list))
        #logging.debug(f"Parsing {input_file}")

        #TODO: filter unique name+file+line loc *here*
        with io.open(filename, encoding='utf-8') as f:
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

                if not collect_passed_remarks and not (isinstance(remark, Missed) |
                                                       isinstance(remark, Failure)):
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

                # If we're reading a back a diff yaml file, max_hotness is
                # already captured which may actually be less than the
                # max hotness found in the file.
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

    def generate_remarks(self, exclude_names='', exclude_text='',
                         collect_passed_remarks=False,
                         annotate_external=False):
        all_remarks, file_remarks, should_display_hotness = \
            self._gather_results(filename=self.file_path,
                                 exclude_names=exclude_names,
                                 exclude_text=exclude_text,
                                 collect_passed_remarks=collect_passed_remarks,
                                 annotate_external=annotate_external)

        self.context = Context()

        for remark in all_remarks.values():
            if isinstance(remark, Passed) and (remark.Pass == "inline"
                                               and remark.Name == "Inlined"):
                for arg in remark.Args:
                    arg_dict = dict(list(arg))
                    caller = arg_dict.get('Caller')
                    if caller:
                        try:
                            self.context.caller_loc[caller] = \
                                arg_dict['DebugLoc']
                        except KeyError:
                            pass

        self.all_remarks = all_remarks
        self.file_remarks = file_remarks
        self.should_display_hotness = should_display_hotness

    def get_remarks(self, remark_type=None, remark_status=None):
        remarks = self.all_remarks.values()

        if remark_type is not None:
            remarks = [r for r in remarks if r.Pass == remark_type]

        assert remark_status in [None, 'Passed', 'Missed', 'Failure']
        if remark_status is not None:
            remarks = [r for r in remarks if r.yaml_tag == '!' + remark_status]

        return remarks

    def __del__(self):
        # Delete the file
        try:
            os.remove(self.file_path)
        except FileNotFoundError:
            pass
