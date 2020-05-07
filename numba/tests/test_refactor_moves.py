import sys
import unittest
import warnings
import subprocess
from importlib import import_module
from contextlib import contextmanager
from types import ModuleType

from numba.core.errors import NumbaDeprecationWarning, _MovedModule
from numba.tests.support import TestCase
from numba.core.utils import PYVERSION

IS_PY36 = PYVERSION[:2] == (3, 6)
NOT_PY36 = not IS_PY36

# All pre0.49 top-level module re-export that will be removed.
_pre_49_top_level_modules = [
    'analysis',
    'annotations',
    'appdirs',
    'array_analysis',
    'bytecode',
    'byteflow',
    'caching',
    'callwrapper',
    'cgutils',
    'charseq',
    'compiler',
    'compiler_lock',
    'compiler_machinery',
    'consts',
    'controlflow',
    # 'ctypes_support', # removed
    'dataflow',
    'datamodel',
    'debuginfo',
    'decorators',
    'dictobject',
    'dispatcher',
    'entrypoints',
    'funcdesc',
    'generators',
    'inline_closurecall',
    'interpreter',
    # 'io_support', # removed
    'ir',
    'ir_utils',
    'itanium_mangler',
    'listobject',
    'lowering',
    'npdatetime',
    # 'npyufunc', # removed
    'numpy_support',
    'object_mode_passes',
    'parfor',
    'postproc',
    'pylowering',
    'pythonapi',
    'rewrites',
    'runtime',
    'serialize',
    'sigutils',
    # 'six', # removed
    'special',
    'stencilparfor',
    # 'targets', # removed
    'tracing',
    'transforms',
    'typeconv',
    'typed_passes',
    'typedobjectutils',
    'typeinfer',
    'typing',
    'unicode',
    'unicode_support',
    'unsafe',
    'untyped_passes',
    'utils',
    'withcontexts',
]


class TestAPIMoves_Q1_2020(TestCase):
    """ Checks API moves made in Q1 2020, this roughly corresponds to 0.48->0.49
    """

    def check_warning(self, from_mod, to_mod):
        """
        Returns a context manager to check that a NumbaDeprecationWarning is
        raised in the context managed block and that the warning contains
        information that from_mod has moved to to_mod
        """
        # make sure the `to_mod` location actually exists (None indicates no
        # replacement"
        if to_mod is not None:
            import_module(to_mod)

        @contextmanager
        def checker(fn=None, skip=False):
            """
            If fn is not None then a check will be made to ensure the module
            level `__getattr__`
            """
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", NumbaDeprecationWarning)
                yield
            if skip:
                return
            self.assertTrue(len(w) > 0)
            _require = "requested from a module that has moved location"
            for x in w:
                if to_mod is not None:
                    c1 = _require in str(x.message)
                    c2 = from_mod in str(x.message)
                    c3 = to_mod in str(x.message)
                else:  # check for help link
                    c1 = True
                    c2 = from_mod in str(x.message)
                    c3 = "gitter.im" in str(x.message)
                if fn is not None:
                    c4 = "Import of '{}' requested".format(fn) in str(
                        x.message
                    )
                else:
                    c4 = True
                if c1 and c2 and c3 and c4:
                    break
            else:
                raise ValueError("No valid warning message found")

        return checker

    def test_nonexistant_moved_raises_not_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", NumbaDeprecationWarning)
            with self.assertRaises(ImportError) as raises:
                from numba import nonexistant_thing  # noqa: F401
            self.assertIn("cannot import name", str(raises.exception))
            self.assertIn("nonexistant_thing", str(raises.exception))
        self.assertEqual(len(w), 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", NumbaDeprecationWarning)
            with self.assertRaises(ImportError) as raises:
                import numba.nonexistant_thing  # noqa: F401
            self.assertIn("No module named", str(raises.exception))
            self.assertIn("nonexistant_thing", str(raises.exception))
        self.assertEqual(len(w), 0)

    def test_numba_utils(self):
        checker = self.check_warning("numba.utils", "numba.core.utils")
        with checker(skip=IS_PY36):
            import numba.utils

        for fn in ("pysignature", "OPERATORS_TO_BUILTINS"):
            with checker(fn):
                getattr(numba.utils, fn)

    def test_numba_untyped_passes(self):
        checker = self.check_warning(
            "numba.untyped_passes", "numba.core.untyped_passes"
        )
        with checker(skip=IS_PY36):
            import numba.untyped_passes

        fn = "InlineClosureLikes"
        with checker(fn):
            getattr(numba.untyped_passes, fn)

    def test_numba_config(self):
        checker = self.check_warning(
            "numba.config", "numba.core.config"
        )
        with checker(skip=IS_PY36):
            import numba.config

        fn = "DISABLE_JIT"
        with checker(fn):
            getattr(numba.config, fn)

    def test_numba_unsafe(self):
        # this is a bit dubious, unsafe was largely split up and moved.
        # numba.unsafe.refcount is in use in external packages and is now in
        # core
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", NumbaDeprecationWarning)
            fn = "refcount"
            import numba.unsafe

            getattr(numba.unsafe, fn)

        if NOT_PY36:
            for x in w:
                if "No direct replacement for 'numba.unsafe'" in str(x.message):
                    break
            else:
                self.fail("Could not find expected warning message")

    def test_numba_unsafe_ndarray(self):
        checker = self.check_warning(
            "numba.unsafe.ndarray", "numba.np.unsafe.ndarray"
        )
        with checker():
            import numba.unsafe.ndarray

        fn = "empty_inferred"
        with checker(fn):
            getattr(numba.unsafe.ndarray, fn)

    def test_numba_unicode(self):
        checker = self.check_warning("numba.unicode", "numba.cpython.unicode")
        with checker(skip=IS_PY36):
            import numba.unicode

        for fn in (
            "unbox_unicode_str",
            "make_string_from_constant",
            "_slice_span",
            "_normalize_slice",
            "_empty_string",
            "PY_UNICODE_1BYTE_KIND",
            "PY_UNICODE_2BYTE_KIND",
            "PY_UNICODE_4BYTE_KIND",
            "PY_UNICODE_WCHAR_KIND",
        ):
            with checker(fn):
                getattr(numba.unicode, fn)

    def test_aaaaa_numba_typing(self):
        # silly 'aaaaa' name to game test ordering, warnings only get triggered
        # once and the `TestAPIMoves_Q1_2020` hits numba.typing.* so this needs
        # to run first
        checker = self.check_warning("numba.typing", "numba.core.typing")

        with checker(skip=IS_PY36):
            import numba.typing

        for fn in (
            "signature",
            "make_concrete_template",
            "Signature",
            "fold_arguments",
        ):
            with checker(fn):
                getattr(numba.typing, fn)

    def test_numba_typing_typeof(self):
        checker = self.check_warning(
            "numba.typing.typeof", "numba.core.typing.typeof"
        )
        with checker():
            import numba.typing.typeof

        for fn in ("typeof_impl", "_typeof_ndarray"):
            with checker(fn):
                getattr(numba.typing.typeof, fn)

    def test_numba_typing_templates(self):
        checker = self.check_warning(
            "numba.typing.templates", "numba.core.typing.templates"
        )
        with checker():
            import numba.typing.templates

        for fn in (
            "signature",
            "infer_global",
            "infer_getattr",
            "infer",
            "bound_function",
            "AttributeTemplate",
            "AbstractTemplate",
        ):
            with checker(fn):
                getattr(numba.typing.templates, fn)

    def test_numba_typing_npydecl(self):
        checker = self.check_warning(
            "numba.typing.npydecl", "numba.core.typing.npydecl"
        )
        with checker():
            import numba.typing.npydecl

        for fn in (
            "supported_ufuncs",
            "NumpyRulesInplaceArrayOperator",
            "NumpyRulesArrayOperator",
            "NdConcatenate",
        ):
            with checker(fn):
                getattr(numba.typing.npydecl, fn)

    def test_numba_typing_ctypes_utils(self):
        checker = self.check_warning(
            "numba.typing.ctypes_utils", "numba.core.typing.ctypes_utils"
        )
        with checker():
            import numba.typing.ctypes_utils

        fn = "make_function_type"
        with checker(fn):
            getattr(numba.typing.ctypes_utils, fn)

    def test_numba_typing_collections(self):
        checker = self.check_warning(
            "numba.typing.collections", "numba.core.typing.collections"
        )
        with checker():
            import numba.typing.collections

        fn = "GetItemSequence"
        with checker(fn):
            getattr(numba.typing.collections, fn)

    def test_numba_typing_builtins(self):
        checker = self.check_warning(
            "numba.typing.builtins", "numba.core.typing.builtins"
        )
        with checker():
            import numba.typing.builtins

        for fn in (
            "MinMaxBase",
            "IndexValueType",
        ):
            with checker(fn):
                getattr(numba.typing.builtins, fn)

    def test_numba_typing_arraydecl(self):
        checker = self.check_warning(
            "numba.typing.arraydecl", "numba.core.typing.arraydecl"
        )
        with checker():
            import numba.typing.arraydecl

        for fn in (
            "get_array_index_type",
            "ArrayAttribute",
        ):
            with checker(fn):
                getattr(numba.typing.arraydecl, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.typing.arraydecl import ArrayAttribute

            ArrayAttribute.resolve_var
            ArrayAttribute.resolve_sum
            ArrayAttribute.resolve_prod

    def test_numba_typeinfer(self):
        checker = self.check_warning("numba.typeinfer", "numba.core.typeinfer")
        with checker(skip=IS_PY36):
            import numba.typeinfer

        fn = "IntrinsicCallConstraint"
        with checker(fn):
            getattr(numba.typeinfer, fn)

    def test_numba_typedobjectutils(self):
        checker = self.check_warning(
            "numba.typedobjectutils", "numba.typed.typedobjectutils"
        )
        with checker(skip=IS_PY36):
            import numba.typedobjectutils

        fn = "_cast"
        with checker(fn):
            getattr(numba.typedobjectutils, fn)

    def test_numba_typed_passes(self):
        checker = self.check_warning(
            "numba.typed_passes", "numba.core.typed_passes"
        )
        with checker(skip=IS_PY36):
            import numba.typed_passes

        for fn in ("type_inference_stage", "AnnotateTypes"):
            with checker(fn):
                getattr(numba.typed_passes, fn)

    def test_numba_typed(self):
        # no refactoring was done to `numba.typed`.
        import numba.typed  # noqa: F401
        from numba.typed import List

        List.empty_list
        from numba.typed import Dict

        Dict.empty

    def test_numba_typeconv(self):
        checker = self.check_warning("numba.typeconv", "numba.core.typeconv")
        with checker(skip=IS_PY36):
            import numba.typeconv

        fn = "Conversion"
        with checker(fn):
            getattr(numba.typeconv, fn)

    def test_aaaaa_numba_targets(self):
        # silly 'aaaaa' name to game test ordering, warnings only get triggered
        # once and the `TestAPIMoves_Q1_2020` hits numba.targets.* so this needs
        # to run first
        checker = self.check_warning("numba.targets", None)
        with checker():
            import numba.targets  # noqa: F401

    def test_numba_targets_boxing(self):
        checker = self.check_warning(
            "numba.targets.boxing", "numba.core.boxing"
        )
        with checker():
            import numba.targets
            numba.targets.boxing

        for fn in ("box_array", "_NumbaTypeHelper"):
            with checker(fn):
                getattr(numba.targets.boxing, fn)

    def test_numba_targets_callconv(self):
        checker = self.check_warning(
            "numba.targets.callconv", "numba.core.callconv"
        )
        with checker():
            import numba.targets
            numba.targets.callconv

        fn = "RETCODE_USEREXC"
        with checker(fn):
            getattr(numba.targets.callconv, fn)

    def test_numba_targets_hashing(self):
        checker = self.check_warning(
            "numba.targets.hashing", "numba.cpython.hashing"
        )
        with checker():
            import numba.targets
            numba.targets.hashing

        # some codebases claim to use these, but they are not present in 0.48:
        # '_Py_HashSecret_siphash_k0', '_Py_HashSecret_siphash_k1',
        # '_Py_HashSecret_djbx33a_suffix'
        fn = "_Py_hash_t"
        with checker(fn):
            getattr(numba.targets.hashing, fn)

    def test_numba_targets_ufunc_db(self):
        checker = self.check_warning(
            "numba.targets.ufunc_db", "numba.np.ufunc_db"
        )
        with checker():
            import numba.targets
            numba.targets.ufunc_db

        fn = "get_ufuncs"
        with checker(fn):
            getattr(numba.targets.ufunc_db, fn)

    def test_numba_targets_slicing(self):
        checker = self.check_warning(
            "numba.targets.slicing", "numba.cpython.slicing"
        )
        with checker():
            import numba.targets
            numba.targets.slicing

        for fn in ("guard_invalid_slice", "get_slice_length", "fix_slice"):
            with checker(fn):
                getattr(numba.targets.slicing, fn)

    def test_numba_targets_setobj(self):
        checker = self.check_warning(
            "numba.targets.setobj", "numba.cpython.setobj"
        )
        with checker():
            import numba.targets
            numba.targets.setobj

        fn = "set_empty_constructor"
        with checker(fn):
            getattr(numba.targets.setobj, fn)

    def test_numba_targets_registry(self):
        checker = self.check_warning(
            "numba.targets.registry", "numba.core.registry"
        )
        with checker():
            import numba.targets
            numba.targets.registry

        for fn in ("dispatcher_registry", "cpu_target"):
            with checker(fn):
                getattr(numba.targets.registry, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.targets.registry import cpu_target

            typing_context = cpu_target.typing_context
            typing_context.resolve_value_type

    def test_numba_targets_options(self):
        checker = self.check_warning(
            "numba.targets.options", "numba.core.options"
        )
        with checker():
            import numba.targets
            numba.targets.options

        fn = "TargetOptions"
        with checker(fn):
            getattr(numba.targets.options, fn)

    def test_numba_targets_listobj(self):
        checker = self.check_warning(
            "numba.targets.listobj", "numba.cpython.listobj"
        )
        with checker():
            import numba.targets
            numba.targets.listobj

        fn = "ListInstance"
        with checker(fn):
            getattr(numba.targets.listobj, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.targets.listobj import ListInstance

            ListInstance.allocate

    def test_numba_targets_imputils(self):
        checker = self.check_warning(
            "numba.targets.imputils", "numba.core.imputils"
        )
        with checker():
            import numba.targets
            numba.targets.imputils

        for fn in (
            "lower_cast",
            "iternext_impl",
            "impl_ret_new_ref",
            "RefType",
        ):
            with checker(fn):
                getattr(numba.targets.imputils, fn)

    def test_numba_targets_cpu(self):
        checker = self.check_warning("numba.targets.cpu", "numba.core.cpu")
        with checker():
            import numba.targets
            numba.targets.cpu

        for fn in ("ParallelOptions", "CPUTargetOptions", "CPUContext"):
            with checker(fn):
                getattr(numba.targets.cpu, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.targets.cpu import CPUTargetOptions

            CPUTargetOptions.OPTIONS

    def test_numba_targets_builtins(self):
        checker = self.check_warning(
            "numba.targets.builtins", "numba.cpython.builtins"
        )
        with checker():
            import numba.targets
            numba.targets.builtins

        for fn in ("get_type_min_value", "get_type_max_value"):
            with checker(fn):
                getattr(numba.targets.builtins, fn)

    def test_numba_targets_arrayobj(self):
        checker = self.check_warning(
            "numba.targets.arrayobj", "numba.np.arrayobj"
        )
        with checker():
            import numba.targets
            numba.targets.arrayobj

        for fn in (
            "store_item",
            "setitem_array",
            "populate_array",
            "numpy_empty_nd",
            "make_array",
            "getiter_array",
            "getitem_arraynd_intp",
            "getitem_array_tuple",
            "fancy_getitem_array",
            "array_reshape",
            "array_ravel",
            "array_len",
            "array_flatten",
        ):
            with checker(fn):
                getattr(numba.targets.arrayobj, fn)

    def test_numba_targets_arraymath(self):
        checker = self.check_warning(
            "numba.targets.arraymath", "numba.np.arraymath"
        )
        with checker():
            import numba.targets
            numba.targets.arraymath

        fn = "get_isnan"
        with checker(fn):
            getattr(numba.targets.arraymath, fn)

    def test_numba_targets_nonexistant(self):
        # not testing for this warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", NumbaDeprecationWarning)
            import numba.targets

        with self.assertRaises(AttributeError) as raises:
            numba.targets.does_not_exist
        exc = raises.exception
        self.assertIn("no attribute or submodule", str(exc))

        # chain shows both the moved attribute lookup and import failing
        self.assertIsInstance(exc.__context__, ModuleNotFoundError)
        self.assertIsInstance(exc.__context__.__context__, AttributeError)

    def test_numba_stencilparfor(self):
        checker = self.check_warning(
            "numba.stencilparfor", "numba.stencils.stencilparfor"
        )
        with checker(skip=IS_PY36):
            import numba.stencilparfor

        fn = "_compute_last_ind"
        with checker(fn):
            getattr(numba.stencilparfor, fn)

    def test_numba_stencil(self):
        checker = self.check_warning("numba.stencil", "numba.stencils.stencil")
        with checker():
            import numba.stencil

        fn = "StencilFunc"
        with checker(fn):
            getattr(numba.stencil, fn)

    def test_numba_runtime_nrt(self):
        checker = self.check_warning(
            "numba.runtime.nrt", "numba.core.runtime.nrt"
        )
        with checker():
            import numba.runtime.nrt

        fn = "rtsys"
        with checker(fn):
            getattr(numba.runtime.nrt, fn)

    def test_numba_runtime(self):
        checker = self.check_warning("numba.runtime", "numba.core.runtime")
        with checker(skip=IS_PY36):
            import numba.runtime

        fn = "nrt"
        with checker(fn):
            getattr(numba.runtime, fn)

    def test_numba_rewrites(self):
        checker = self.check_warning("numba.rewrites", "numba.core.rewrites")
        with checker(skip=IS_PY36):
            import numba.rewrites

        fn = "rewrite_registry"
        with checker(skip=IS_PY36):
            getattr(numba.rewrites, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.rewrites import rewrite_registry

            rewrite_registry.apply

    def test_numba_pythonapi(self):
        checker = self.check_warning("numba.pythonapi", "numba.core.pythonapi")
        with checker(skip=IS_PY36):
            import numba.pythonapi

        for fn in (
            "_UnboxContext",
            "_BoxContext",
        ):
            with checker(fn):
                getattr(numba.pythonapi, fn)

    def test_numba_parfor(self):
        checker = self.check_warning("numba.parfor", "numba.parfors.parfor")
        with checker(skip=IS_PY36):
            import numba.parfor

        for fn in (
            "replace_functions_map",
            "min_checker",
            "maximize_fusion",
            "max_checker",
            "lower_parfor_sequential",
            "internal_prange",
            "init_prange",
            "argmin_checker",
            "argmax_checker",
            "PreParforPass",
            "ParforPass",
            "Parfor",
        ):
            with checker(fn):
                getattr(numba.parfor, fn)

    def test_numba_numpy_support(self):
        checker = self.check_warning(
            "numba.numpy_support", "numba.np.numpy_support"
        )
        if NOT_PY36:
            with checker():
                import numba.numpy_support
        import numba
        for fn in ("map_layout", "from_dtype", "as_dtype"):
            with checker(fn):
                getattr(numba.numpy_support, fn)

    def test_numba_targets_npdatetime(self):
        checker = self.check_warning(
            "numba.targets.npdatetime", "numba.np.npdatetime"
        )
        with checker():
            import numba.targets
            numba.targets.npdatetime

        for fn in ("convert_datetime_for_arith",):
            with checker(fn):
                getattr(numba.targets.npdatetime, fn)

    def test_numba_npdatetime(self):
        #  the <= 0.48  numba.npdatetime module provides the datetime helper
        # functions, this got moved in location and name
        checker = self.check_warning(
            "numba.npdatetime", "numba.np.npdatetime_helpers"
        )
        with checker(skip=IS_PY36):
            import numba.npdatetime

        for fn in ("DATETIME_UNITS",):
            with checker(fn):
                getattr(numba.npdatetime, fn)

    def test_numba_lowering(self):
        checker = self.check_warning("numba.lowering", "numba.core.lowering")
        with checker(skip=IS_PY36):
            import numba.lowering  # noqa: F401

    def test_numba_ir_utils(self):
        checker = self.check_warning("numba.ir_utils", "numba.core.ir_utils")
        with checker(skip=IS_PY36):
            import numba.ir_utils

        for fn in (
            "simplify_CFG",
            "replace_vars_stmt",
            "replace_arg_nodes",
            "remove_dead",
            "remove_call_handlers",
            "next_label",
            "mk_unique_var",
            "get_ir_of_code",
            "get_definition",
            "find_const",
            "dprint_func_ir",
            "compute_cfg_from_blocks",
            "compile_to_numba_ir",
            "build_definitions",
            "alias_func_extensions",
            "_max_label",
            "_add_alias",
            "GuardException",
        ):
            with checker(fn):
                getattr(numba.ir_utils, fn)

    def test_numba_ir(self):
        checker = self.check_warning("numba.ir", "numba.core.ir")
        with checker(skip=IS_PY36):
            import numba.ir

        for fn in ("Assign", "Const", "Expr", "Var"):
            with checker(fn):
                getattr(numba.ir, fn)

    def test_numba_inline_closurecall(self):
        checker = self.check_warning(
            "numba.inline_closurecall", "numba.core.inline_closurecall"
        )
        with checker(skip=IS_PY36):
            import numba.inline_closurecall

        for fn in (
            "inline_closure_call",
            "_replace_returns",
            "_replace_freevars",
            "_add_definitions",
        ):
            with checker(fn):
                getattr(numba.inline_closurecall, fn)

    def test_numba_errors(self):
        checker = self.check_warning("numba.errors", "numba.core.errors")
        with checker(skip=IS_PY36):
            import numba.errors

        for fn in (
            "WarningsFixer",
            "TypingError",
            "NumbaWarning",
            "ForceLiteralArg",
        ):
            with checker(fn):
                getattr(numba.errors, fn)

    def test_numba_dispatcher(self):
        checker = self.check_warning(
            "numba.dispatcher", "numba.core.dispatcher"
        )
        with checker(skip=IS_PY36):
            import numba.dispatcher

        for fn in (
            "ObjModeLiftedWith",
            "Dispatcher",
        ):
            with checker(fn):
                getattr(numba.dispatcher, fn)

    def test_numba_dictobject(self):
        checker = self.check_warning(
            "numba.dictobject", "numba.typed.dictobject"
        )
        with checker(skip=IS_PY36):
            import numba.dictobject

        fn = "DictModel"
        with checker(fn):
            getattr(numba.dictobject, fn)

    def test_numba_datamodel(self):
        checker = self.check_warning("numba.datamodel", "numba.core.datamodel")
        with checker(skip=IS_PY36):
            import numba.datamodel

        for fn in (
            "registry",
            "register_default",
        ):
            with checker(fn):
                getattr(numba.datamodel, fn)

        checker = self.check_warning(
            "numba.datamodel.models", "numba.core.datamodel.models"
        )
        with checker():
            import numba.datamodel.models

        for fn in ("StructModel", "PointerModel", "BooleanModel"):
            with checker(fn):
                getattr(numba.datamodel.models, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.datamodel import registry

            registry.register_default

    def test_numba_compiler_machinery(self):
        checker = self.check_warning(
            "numba.compiler_machinery", "numba.core.compiler_machinery"
        )
        with checker(skip=IS_PY36):
            import numba.compiler_machinery  # noqa: F401

    def test_numba_compiler(self):
        checker = self.check_warning("numba.compiler", "numba.core.compiler")
        with checker(skip=IS_PY36):
            import numba.compiler

        for fn in (
            "run_frontend",
            "StateDict",
            "Flags",
            "DefaultPassBuilder",
            "CompilerBase",
        ):
            with checker(fn):
                getattr(numba.compiler, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.compiler import DefaultPassBuilder, Flags

            DefaultPassBuilder.define_nopython_pipeline
            Flags.OPTIONS

    def test_numba_cgutils(self):
        checker = self.check_warning("numba.cgutils", "numba.core.cgutils")
        with checker(skip=IS_PY36):
            import numba.cgutils
        for fn in (
            "unpack_tuple",
            "true_bit",
            "is_not_null",
            "increment_index",
            "get_null_value",
            "false_bit",
            "create_struct_proxy",
            "alloca_once_value",
            "alloca_once",
        ):
            with checker(fn):
                getattr(numba.cgutils, fn)

    def test_numba_array_analysis(self):
        checker = self.check_warning(
            "numba.array_analysis", "numba.parfors.array_analysis"
        )
        with checker(skip=IS_PY36):
            import numba.array_analysis

        for fn in ("ArrayAnalysis", "array_analysis_extensions"):
            with checker(fn):
                getattr(numba.array_analysis, fn)

        # want to check this works, not that it warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaDeprecationWarning)
            from numba.array_analysis import ArrayAnalysis

            args = (None,) * 4
            ArrayAnalysis(*args)._analyze_op_static_getitem
            ArrayAnalysis(*args)._analyze_broadcast

    def test_numba_analysis(self):
        checker = self.check_warning("numba.analysis", "numba.core.analysis")
        if NOT_PY36:
            with checker():
                import numba.analysis
        import numba
        for fn in (
            "ir_extension_usedefs",
            "compute_use_defs",
            "compute_cfg_from_blocks",
            "_use_defs_result",
        ):
            with checker(fn):
                getattr(numba.analysis, fn)

    def test_numba_decorators(self):
        checker = self.check_warning(
            "numba.decorators", "numba.core.decorators"
        )
        with checker(skip=IS_PY36):
            import numba.decorators
        for fn in (
            "njit",
            "jit",
            "generated_jit",
        ):
            with checker(fn):
                getattr(numba.decorators, fn)

    def test_numba_jitclass(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", NumbaDeprecationWarning)
            from numba import jitclass # SHOULD NOT WARN
        for x in w:
            if "numba.jitclass" in str(x.message):
                raise ValueError("Unexpected warning encountered")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", NumbaDeprecationWarning)
            @jitclass(spec=[])
            class foo:
                def __init__(self):
                    pass
            foo()
        for x in w:
            if "has moved to 'numba.experimental.jitclass'" in str(x.message):
                break
        else:
            raise ValueError("Expected warning not found")

    def run_in_fresh_python(self, code):
        return subprocess.check_output(
            [sys.executable, '-c', code],
            stderr=subprocess.STDOUT,
        ).decode()

    def test_top_level_export(self):
        def compare_mod(left, right):
            if isinstance(left, _MovedModule):
                self.assertIsInstance(right, ModuleType)
            elif isinstance(right, _MovedModule):
                self.assertIsInstance(left, ModuleType)
            else:
                self.assertIs(left, right)

        for mod in _pre_49_top_level_modules:
            if mod in {'unsafe'}:
                # SKIP.
                # different behavior and tested elsewhere.
                continue
            oldmodname = f'numba.{mod}'
            with self.subTest(oldmodname):
                oldmod = import_module(oldmodname)
                # Make sure the target module is available
                newmodname = oldmod._MovedModule__new_module
                newmod = import_module(newmodname)
                for k in dir(newmod):
                    if k.startswith('_'):
                        continue
                    obj = getattr(oldmod, k)
                    compare_mod(getattr(newmod, k), obj)

    def test_top_level_export_as_attr(self):
        for mod in _pre_49_top_level_modules:
            mod = f'numba.{mod}'
            with self.subTest(mod):
                out = self.run_in_fresh_python(f'import numba; {mod}')
                if NOT_PY36:
                    self.assertIn("NumbaDeprecationWarning", out)
                    self.assertIn(f"Import requested from: '{mod}'", out)

    def test_top_level_export_as_import(self):
        for mod in _pre_49_top_level_modules:
            if mod in {'unsafe'}:
                # SKIP.
                # different behavior and tested elsewhere.
                continue
            mod = f'numba.{mod}'
            with self.subTest(mod):
                out = self.run_in_fresh_python(f'import {mod}')
                if NOT_PY36:
                    self.assertIn("NumbaDeprecationWarning", out)
                    self.assertIn(f"Import requested from: '{mod}'", out)

    def test_top_level_export_attr_matches_import(self):
        for mod in _pre_49_top_level_modules:
            mod = f'numba.{mod}'
            with self.subTest(mod):
                code = [
                    'import numba',
                    f'import {mod} as ref',
                    f'assert {mod} is ref',
                ]
                self.run_in_fresh_python('; '.join(code))

    def test_issue_5528(self):
        # Check specific issues from the ticket.
        code = """
import numba
numba.numpy_support.from_dtype
"""
        self.run_in_fresh_python(code)

        code = """
import numba.numpy_support
numba.numpy_support.from_dtype
"""
        self.run_in_fresh_python(code)

        code = """
from numba.numpy_support import from_dtype
"""
        self.run_in_fresh_python(code)

        code = """
from numba.targets import quicksort
"""
        self.run_in_fresh_python(code)

        code = """
import numba
numba.special
"""
        self.run_in_fresh_python(code)

        code = """
from numba.special import literally
"""
        self.run_in_fresh_python(code)

        code = """
from numba.special import prange
"""
        self.run_in_fresh_python(code)

        code = """
from numba import postproc
"""
        self.run_in_fresh_python(code)

        code = """
import numba
numba.analysis.ir_extension_usedefs

"""
        self.run_in_fresh_python(code)

        code = """
from numba import consts
"""
        self.run_in_fresh_python(code)

        code = """
import numba
numba.extending.lower_builtin

"""
        self.run_in_fresh_python(code)

        code = """
from numba.config import IS_32BITS
"""
        self.run_in_fresh_python(code)

        code = """
import numba
numba.datamodel.register_default
"""
        self.run_in_fresh_python(code)


if __name__ == "__main__":
    unittest.main()
