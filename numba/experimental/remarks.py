import yaml
import re
import time
import os

from itanium_demangler import parse as demangle

from numba.core import config


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
        filename = '{:08x}-{}.yaml'.format(id(self), time.time())
        remarks_file = os.path.join(dirpath, filename)

        return remarks_file
