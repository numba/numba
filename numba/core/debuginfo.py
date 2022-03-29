"""
Implements helpers to build LLVM debuginfo.
"""


import abc
import os.path
from contextlib import contextmanager

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config


@contextmanager
def suspend_emission(builder):
    """Suspends the emission of debug_metadata for the duration of the context
    managed block."""
    ref = builder.debug_metadata
    builder.debug_metadata = None
    try:
        yield
    finally:
        builder.debug_metadata = ref


class AbstractDIBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def mark_variable(self, builder, allocavalue, name, lltype, size, line,
                      datamodel=None, argidx=None):
        """Emit debug info for the variable.
        """
        pass

    @abc.abstractmethod
    def mark_location(self, builder, line):
        """Emit source location information to the given IRBuilder.
        """
        pass

    @abc.abstractmethod
    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        """Emit source location information for the given function.
        """
        pass

    @abc.abstractmethod
    def initialize(self):
        """Initialize the debug info. An opportunity for the debuginfo to
        prepare any necessary data structures.
        """

    @abc.abstractmethod
    def finalize(self):
        """Finalize the debuginfo by emitting all necessary metadata.
        """
        pass


class DummyDIBuilder(AbstractDIBuilder):

    def __init__(self, module, filepath, cgctx):
        pass

    def mark_variable(self, builder, allocavalue, name, lltype, size, line,
                      datamodel=None, argidx=None):
        pass

    def mark_location(self, builder, line):
        pass

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass


_BYTE_SIZE = 8


class DIBuilder(AbstractDIBuilder):
    DWARF_VERSION = 4
    DEBUG_INFO_VERSION = 3
    DBG_CU_NAME = 'llvm.dbg.cu'
    _DEBUG = False

    def __init__(self, module, filepath, cgctx):
        self.module = module
        self.filepath = os.path.abspath(filepath)
        self.difile = self._di_file()
        self.subprograms = []
        self.cgctx = cgctx
        self.initialize()

    def initialize(self):
        # Create the compile unit now because it is referenced when
        # constructing subprograms
        self.dicompileunit = self._di_compile_unit()

    def _var_type(self, lltype, size, datamodel=None):
        if self._DEBUG:
            print("-->", lltype, size, datamodel,
                  getattr(datamodel, 'fe_type', 'NO FE TYPE'))
        m = self.module
        bitsize = _BYTE_SIZE * size

        int_type = ir.IntType,
        real_type = ir.FloatType, ir.DoubleType
        # For simple numeric types, choose the closest encoding.
        # We treat all integers as unsigned when there's no known datamodel.
        if isinstance(lltype, int_type + real_type):
            if datamodel is None:
                # This is probably something like an `i8*` member of a struct
                name = str(lltype)
                if isinstance(lltype, int_type):
                    ditok = 'DW_ATE_unsigned'
                else:
                    ditok = 'DW_ATE_float'
            else:
                # This is probably a known int/float scalar type
                name = str(datamodel.fe_type)
                if isinstance(datamodel.fe_type, types.Integer):
                    if datamodel.fe_type.signed:
                        ditok = 'DW_ATE_signed'
                    else:
                        ditok = 'DW_ATE_unsigned'
                else:
                    ditok = 'DW_ATE_float'
            mdtype = m.add_debug_info('DIBasicType', {
                'name': name,
                'size': bitsize,
                'encoding': ir.DIToken(ditok),
            })
        elif isinstance(datamodel, ComplexModel):
            # TODO: Is there a better way of determining "this is a complex
            # number"?
            #
            # NOTE: Commented below is the way to generate the metadata for a
            # C99 complex type that's directly supported by DWARF. Numba however
            # generates a struct with real/imag cf. CPython to give a more
            # pythonic feel to inspection.
            #
            # mdtype = m.add_debug_info('DIBasicType', {
            #  'name': f"{datamodel.fe_type} ({str(lltype)})",
            #  'size': bitsize,
            # 'encoding': ir.DIToken('DW_ATE_complex_float'),
            #})
            meta = []
            offset = 0
            for ix, name in enumerate(('real', 'imag')):
                component = lltype.elements[ix]
                component_size = self.cgctx.get_abi_sizeof(component)
                component_basetype = m.add_debug_info('DIBasicType', {
                    'name': str(component),
                    'size': _BYTE_SIZE * component_size, # bits
                    'encoding': ir.DIToken('DW_ATE_float'),
                })
                derived_type = m.add_debug_info('DIDerivedType', {
                    'tag': ir.DIToken('DW_TAG_member'),
                    'name': name,
                    'baseType': component_basetype,
                    'size': _BYTE_SIZE * component_size, # DW_TAG_member size is in bits
                    'offset': offset,
                })
                meta.append(derived_type)
                offset += (_BYTE_SIZE * component_size) # offset is in bits
            mdtype = m.add_debug_info('DICompositeType', {
                'tag': ir.DIToken('DW_TAG_structure_type'),
                'name': f"{datamodel.fe_type} ({str(lltype)})",
                'identifier': str(lltype),
                'elements': m.add_metadata(meta),
                'size': offset,
            }, is_distinct=True)
        elif isinstance(datamodel, UniTupleModel):
            element = lltype.element
            el_size = self.cgctx.get_abi_sizeof(element)
            basetype = self._var_type(element, el_size)
            name = f"{datamodel.fe_type} ({str(lltype)})"
            count = size // el_size
            mdrange = m.add_debug_info('DISubrange', {
                'count': count,
            })
            mdtype = m.add_debug_info('DICompositeType', {
                'tag': ir.DIToken('DW_TAG_array_type'),
                'baseType': basetype,
                'name': name,
                'size': bitsize,
                'identifier': str(lltype),
                'elements': m.add_metadata([mdrange]),
            })
        elif isinstance(lltype, ir.PointerType):
            model = getattr(datamodel, '_pointee_model', None)
            basetype = self._var_type(lltype.pointee,
                                      self.cgctx.get_abi_sizeof(lltype.pointee),
                                      model)
            mdtype = m.add_debug_info('DIDerivedType', {
                'tag': ir.DIToken('DW_TAG_pointer_type'),
                'baseType': basetype,
                'size': _BYTE_SIZE * self.cgctx.get_abi_sizeof(lltype)
            })
        elif isinstance(lltype, ir.LiteralStructType):
            # Struct type
            meta = []
            offset = 0
            if datamodel is None or not datamodel.inner_models():
                name = f"Anonymous struct ({str(lltype)})"
                for field_id, element in enumerate(lltype.elements):
                    size = self.cgctx.get_abi_sizeof(element)
                    basetype = self._var_type(element, size)
                    derived_type = m.add_debug_info('DIDerivedType', {
                        'tag': ir.DIToken('DW_TAG_member'),
                        'name': f'<field {field_id}>',
                        'baseType': basetype,
                        'size': _BYTE_SIZE * size, # DW_TAG_member size is in bits
                        'offset': offset,
                    })
                    meta.append(derived_type)
                    offset += (_BYTE_SIZE * size) # offset is in bits
            else:
                name = f"{datamodel.fe_type} ({str(lltype)})"
                for element, field, model in zip(lltype.elements,
                                                 datamodel._fields,
                                                 datamodel.inner_models()):
                    size = self.cgctx.get_abi_sizeof(element)
                    basetype = self._var_type(element, size, datamodel=model)
                    derived_type = m.add_debug_info('DIDerivedType', {
                        'tag': ir.DIToken('DW_TAG_member'),
                        'name': field,
                        'baseType': basetype,
                        'size': _BYTE_SIZE * size, # DW_TAG_member size is in bits
                        'offset': offset,
                    })
                    meta.append(derived_type)
                    offset += (_BYTE_SIZE * size) # offset is in bits

            mdtype = m.add_debug_info('DICompositeType', {
                'tag': ir.DIToken('DW_TAG_structure_type'),
                'name': name,
                'identifier': str(lltype),
                'elements': m.add_metadata(meta),
                'size': offset,
            }, is_distinct=True)
        elif isinstance(lltype, ir.ArrayType):
            element = lltype.element
            el_size = self.cgctx.get_abi_sizeof(element)
            basetype = self._var_type(element, el_size)
            count = size // el_size
            mdrange = m.add_debug_info('DISubrange', {
                'count': count,
            })
            mdtype = m.add_debug_info('DICompositeType', {
                'tag': ir.DIToken('DW_TAG_array_type'),
                'baseType': basetype,
                'name': str(lltype),
                'size': bitsize,
                'identifier': str(lltype),
                'elements': m.add_metadata([mdrange]),
            })
        else:
            # For all other types, describe it as sequence of bytes
            count = size
            mdrange = m.add_debug_info('DISubrange', {
                'count': count,
            })
            mdbase = m.add_debug_info('DIBasicType', {
                'name': 'byte',
                'size': _BYTE_SIZE,
                'encoding': ir.DIToken('DW_ATE_unsigned_char'),
            })
            mdtype = m.add_debug_info('DICompositeType', {
                'tag': ir.DIToken('DW_TAG_array_type'),
                'baseType': mdbase,
                'name': str(lltype),
                'size': bitsize,
                'identifier': str(lltype),
                'elements': m.add_metadata([mdrange]),
            })

        return mdtype

    def mark_variable(self, builder, allocavalue, name, lltype, size, line,
                      datamodel=None, argidx=None):

        arg_index = 0 if argidx is None else argidx
        m = self.module
        fnty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
        decl = cgutils.get_or_insert_function(m, fnty, 'llvm.dbg.declare')

        mdtype = self._var_type(lltype, size, datamodel=datamodel)
        name = name.replace('.', '$')    # for gdb to work correctly
        mdlocalvar = m.add_debug_info('DILocalVariable', {
            'name': name,
            'arg': arg_index,
            'scope': self.subprograms[-1],
            'file': self.difile,
            'line': line,
            'type': mdtype,
        })
        mdexpr = m.add_debug_info('DIExpression', {})

        return builder.call(decl, [allocavalue, mdlocalvar, mdexpr])

    def mark_location(self, builder, line):
        builder.debug_metadata = self._add_location(line)

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        name = qualname
        argmap = dict(zip(argnames, argtypes))
        di_subp = self._add_subprogram(name=name, linkagename=function.name,
                                       line=line, function=function,
                                       argmap=argmap)
        function.set_metadata("dbg", di_subp)

        # Don't marked alwaysinline functions as noinline.
        if 'alwaysinline' not in function.attributes:
            # disable inlining for this function for easier debugging
            function.attributes.add('noinline')

    def finalize(self):
        dbgcu = cgutils.get_or_insert_named_metadata(self.module, self.DBG_CU_NAME)
        dbgcu.add(self.dicompileunit)
        self._set_module_flags()

    #
    # Internal APIs
    #

    def _set_module_flags(self):
        """Set the module flags metadata
        """
        module = self.module
        mflags = cgutils.get_or_insert_named_metadata(module, 'llvm.module.flags')
        # Set *require* behavior to warning
        # See http://llvm.org/docs/LangRef.html#module-flags-metadata
        require_warning_behavior = self._const_int(2)
        if self.DWARF_VERSION is not None:
            dwarf_version = module.add_metadata([
                require_warning_behavior,
                "Dwarf Version",
                self._const_int(self.DWARF_VERSION)
            ])
            if dwarf_version not in mflags.operands:
                mflags.add(dwarf_version)
        debuginfo_version = module.add_metadata([
            require_warning_behavior,
            "Debug Info Version",
            self._const_int(self.DEBUG_INFO_VERSION)
        ])
        if debuginfo_version not in mflags.operands:
            mflags.add(debuginfo_version)

    def _add_subprogram(self, name, linkagename, line, function, argmap):
        """Emit subprogram metadata
        """
        subp = self._di_subprogram(name, linkagename, line, function, argmap)
        self.subprograms.append(subp)
        return subp

    def _add_location(self, line):
        """Emit location metatdaa
        """
        loc = self._di_location(line)
        return loc

    @classmethod
    def _const_int(cls, num, bits=32):
        """Util to create constant int in metadata
        """
        return ir.IntType(bits)(num)

    @classmethod
    def _const_bool(cls, boolean):
        """Util to create constant boolean in metadata
        """
        return ir.IntType(1)(boolean)

    #
    # Helpers to emit the metadata nodes
    #

    def _di_file(self):
        return self.module.add_debug_info('DIFile', {
            'directory': os.path.dirname(self.filepath),
            'filename': os.path.basename(self.filepath),
        })

    def _di_compile_unit(self):
        return self.module.add_debug_info('DICompileUnit', {
            'language': ir.DIToken('DW_LANG_C_plus_plus'),
            'file': self.difile,
            # Numba has to pretend to be clang to ensure the prologue is skipped
            # correctly in gdb. See:
            # https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/amd64-tdep.c;h=e563d369d8cb3eb3c2f732c2fa850ec70ba8d63b;hb=a4b0231e179607e47b1cdf1fe15c5dc25e482fad#l2521
            # Note the "producer_is_llvm" call to specialise the prologue
            # handling, this is defined here:
            # https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/producer.c;h=cdfd80d904c09394febd18749bb90359b2d128cc;hb=a4b0231e179607e47b1cdf1fe15c5dc25e482fad#l124
            # and to get a match for this condition the 'producer' must start
            # with "clang ", hence the following...
            'producer': 'clang (Numba)',
            'runtimeVersion': 0,
            'isOptimized': config.OPT != 0,
            'emissionKind': 1,  # 0-NoDebug, 1-FullDebug
        }, is_distinct=True)

    def _di_subroutine_type(self, line, function, argmap):
        # The function call conv needs encoding.
        llfunc = function
        md = []

        for idx, llarg in enumerate(llfunc.args):
            if not llarg.name.startswith('arg.'):
                name = llarg.name.replace('.', '$')    # for gdb to work correctly
                lltype = llarg.type
                size = self.cgctx.get_abi_sizeof(lltype)
                mdtype = self._var_type(lltype, size, datamodel=None)
                md.append(mdtype)

        for idx, (name, nbtype) in enumerate(argmap.items()):
            name = name.replace('.', '$')    # for gdb to work correctly
            datamodel = self.cgctx.data_model_manager[nbtype]
            lltype = self.cgctx.get_value_type(nbtype)
            size = self.cgctx.get_abi_sizeof(lltype)
            mdtype = self._var_type(lltype, size, datamodel=datamodel)
            md.append(mdtype)

        return self.module.add_debug_info('DISubroutineType', {
            'types': self.module.add_metadata(md),
        })

    def _di_subprogram(self, name, linkagename, line, function, argmap):
        return self.module.add_debug_info('DISubprogram', {
            'name': name,
            'linkageName': linkagename,
            'scope': self.difile,
            'file': self.difile,
            'line': line,
            'type': self._di_subroutine_type(line, function, argmap),
            'isLocal': False,
            'isDefinition': True,
            'scopeLine': line,
            'isOptimized': config.OPT != 0,
            'unit': self.dicompileunit,
        }, is_distinct=True)

    def _di_location(self, line):
        return self.module.add_debug_info('DILocation', {
            'line': line,
            'column': 1,
            'scope': self.subprograms[-1],
        })


class NvvmDIBuilder(DIBuilder):
    """
    Only implemented the minimal metadata to get line number information.
    See http://llvm.org/releases/3.4/docs/LangRef.html
    """
    # These constants are copied from llvm3.4
    DW_LANG_Python = 0x0014
    DI_Compile_unit = 786449
    DI_Subroutine_type = 786453
    DI_Subprogram = 786478
    DI_File = 786473

    DWARF_VERSION = None  # don't emit DWARF version
    DEBUG_INFO_VERSION = 1  # as required by NVVM IR Spec
    # Rename DIComputeUnit MD to hide it from llvm.parse_assembly()
    # which strips invalid/outdated debug metadata
    DBG_CU_NAME = 'numba.llvm.dbg.cu'

    # Default member
    # Used in mark_location to remember last lineno to avoid duplication
    _last_lineno = None

    def mark_variable(self, builder, allocavalue, name, lltype, size, line,
                      datamodel=None, argidx=None):
        # unsupported
        pass

    def mark_location(self, builder, line):
        # Avoid duplication
        if self._last_lineno == line:
            return
        self._last_lineno = line
        # Add call to an inline asm to mark line location
        asmty = ir.FunctionType(ir.VoidType(), [])
        asm = ir.InlineAsm(asmty, "// dbg {}".format(line), "",
                           side_effect=True)
        call = builder.call(asm, [])
        md = self._di_location(line)
        call.set_metadata('numba.dbg', md)

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        argmap = dict(zip(argnames, argtypes))
        self._add_subprogram(name=qualname, linkagename=function.name,
                             line=line)

    def _add_subprogram(self, name, linkagename, line):
        """Emit subprogram metadata
        """
        subp = self._di_subprogram(name, linkagename, line)
        self.subprograms.append(subp)
        return subp

    #
    # Helper methods to create the metadata nodes.
    #

    def _filepair(self):
        return self.module.add_metadata([
            os.path.basename(self.filepath),
            os.path.dirname(self.filepath),
        ])

    def _di_file(self):
        return self.module.add_metadata([
            self._const_int(self.DI_File),
            self._filepair(),
        ])

    def _di_compile_unit(self):
        filepair = self._filepair()
        empty = self.module.add_metadata([self._const_int(0)])
        sp_metadata = self.module.add_metadata(self.subprograms)
        return self.module.add_metadata([
            self._const_int(self.DI_Compile_unit),         # tag
            filepair,                   # source directory and file pair
            self._const_int(self.DW_LANG_Python),  # language
            'Numba',                     # producer
            self._const_bool(True),      # optimized
            "",                          # flags??
            self._const_int(0),          # runtime version
            empty,                       # enums types
            empty,                       # retained types
            self.module.add_metadata(self.subprograms),  # subprograms
            empty,                       # global variables
            empty,                       # imported entities
            "",                          # split debug filename
        ])

    def _di_subroutine_type(self):
        types = self.module.add_metadata([None])
        return self.module.add_metadata([
            self._const_int(self.DI_Subroutine_type),                # tag
            self._const_int(0),
            None,
            "",
            self._const_int(0),                 # line of definition
            self._const_int(0, 64),             # size in bits
            self._const_int(0, 64),             # offset in bits
            self._const_int(0, 64),             # align in bits
            self._const_int(0),                 # flags
            None,
            types,
            self._const_int(0),
            None,
            None,
            None,
        ])

    def _di_subprogram(self, name, linkagename, line):
        function_ptr = self.module.get_global(linkagename)
        subroutine_type = self._di_subroutine_type()
        funcvars = self.module.add_metadata([self._const_int(0)])
        context = self._di_file()
        return self.module.add_metadata([
            self._const_int(self.DI_Subprogram),   # tag
            self._filepair(),          # source dir & file
            context,                   # context descriptor
            name,                      # name
            name,                      # display name
            linkagename,               # linkage name
            self._const_int(line),     # line
            subroutine_type,           # type descriptor
            self._const_bool(False),   # is local
            self._const_bool(True),    # is definition
            self._const_int(0),        # virtuality
            self._const_int(0),        # virtual function index
            None,                     # vtable base type
            self._const_int(0),        # flags
            self._const_bool(True),    # is optimized
            function_ptr,              # pointer to function
            None,                      # function template parameters
            None,                      # function declaration descriptor
            funcvars,                     # function variables
            self._const_int(line)      # scope line
        ])

    def _di_location(self, line):
        return self.module.add_metadata([
            self._const_int(line),   # line
            self._const_int(0),      # column
            self.subprograms[-1],    # scope
            None,                    # original scope
        ])

    def initialize(self):
        pass

    def finalize(self):
        # We create the compile unit at this point because subprograms is
        # populated and can be referred to by the compile unit.
        self.dicompileunit = self._di_compile_unit()
        super().finalize()
