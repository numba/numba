Version 0.57.1 (21 June, 2023)
------------------------------

Pull-Requests:

* PR `#8964 <https://github.com/numba/numba/pull/8964>`_: fix missing nopython keyword in cuda random module (`esc <https://github.com/esc>`_)
* PR `#8965 <https://github.com/numba/numba/pull/8965>`_: fix return dtype for np.angle (`guilhermeleobas <https://github.com/guilhermeleobas>`_ `esc <https://github.com/esc>`_)
* PR `#8982 <https://github.com/numba/numba/pull/8982>`_: Don't do the parfor diagnostics pass for the parfor gufunc. (`DrTodd13 <https://github.com/DrTodd13>`_)
* PR `#8996 <https://github.com/numba/numba/pull/8996>`_: adding a test for 8940 (`esc <https://github.com/esc>`_)
* PR `#8958 <https://github.com/numba/numba/pull/8958>`_: resurrect the import, this time in the registry initialization (`esc <https://github.com/esc>`_)
* PR `#8947 <https://github.com/numba/numba/pull/8947>`_: Introduce internal _isinstance_no_warn (`guilhermeleobas <https://github.com/guilhermeleobas>`_ `esc <https://github.com/esc>`_)
* PR `#8998 <https://github.com/numba/numba/pull/8998>`_: Fix 8939 (second attempt) (`esc <https://github.com/esc>`_)
* PR `#8978 <https://github.com/numba/numba/pull/8978>`_: Import MVC packages when using MVCLinker. (`bdice <https://github.com/bdice>`_)
* PR `#8895 <https://github.com/numba/numba/pull/8895>`_: CUDA: Enable caching functions that use CG (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8976 <https://github.com/numba/numba/pull/8976>`_: Fix index URL for ptxcompiler/cubinlinker packages. (`bdice <https://github.com/bdice>`_)
* PR `#9004 <https://github.com/numba/numba/pull/9004>`_: Skip MVC test when libraries unavailable (`gmarkall <https://github.com/gmarkall>`_ `esc <https://github.com/esc>`_)
* PR `#9006 <https://github.com/numba/numba/pull/9006>`_: link to version support table instead of using explicit versions (`esc <https://github.com/esc>`_)
* PR `#9005 <https://github.com/numba/numba/pull/9005>`_: Fix: Issue #8923 - avoid spurious device-to-host transfers in CUDA ufuncs (`gmarkall <https://github.com/gmarkall>`_)


Authors:

* `bdice <https://github.com/bdice>`_
* `DrTodd13 <https://github.com/DrTodd13>`_
* `esc <https://github.com/esc>`_
* `gmarkall <https://github.com/gmarkall>`_

Version 0.57.0 (1 May, 2023)
----------------------------

This release continues to add new features, bug fixes and stability improvements
to Numba. Please note that this release contains a significant number of both
deprecation and pending-deprecation notices with view of making it easier to
develop new technology for Numba in the future. Also note that this will be the
last release to support Windows 32-bit packages produced by the Numba team.

Highlights of core dependency upgrades:

* Support for Python 3.11 (minimum is moved to 3.8)
* Support for NumPy 1.24 (minimum is moved to 1.21)

Python language support enhancements:

* Exception classes now support arguments that are not compile time constant.
* The built-in functions ``hasattr`` and ``getattr`` are supported for compile
  time constant attributes.
* The built-in functions ``str`` and ``repr`` are now implemented similarly to
  their Python implementations. Custom ``__str__`` and ``__repr__``
  functions can be associated with types and work as expected.
* Numba's unicode functionality in ``str.startswith`` now supports kwargs
  ``start`` and ``end``.
* ``min`` and ``max`` now support boolean types.
* Support is added for the ``dict(iterable)`` constructor.

NumPy features/enhancements:

* The largest set of new features is within the ``numpy.random.Generator``
  support, the vast majority of commonly used distributions are now supported.
  Namely:

  * ``Generator.beta``
  * ``Generator.chisquare``
  * ``Generator.exponential``
  * ``Generator.f``
  * ``Generator.gamma``
  * ``Generator.geometric``
  * ``Generator.integers``
  * ``Generator.laplace``
  * ``Generator.logistic``
  * ``Generator.lognormal``
  * ``Generator.logseries``
  * ``Generator.negative_binomial``
  * ``Generator.noncentral_chisquare``
  * ``Generator.noncentral_f``
  * ``Generator.normal``
  * ``Generator.pareto``
  * ``Generator.permutation``
  * ``Generator.poisson``
  * ``Generator.power``
  * ``Generator.random``
  * ``Generator.rayleigh``
  * ``Generator.shuffle``
  * ``Generator.standard_cauchy``
  * ``Generator.standard_exponential``
  * ``Generator.standard_gamma``
  * ``Generator.standard_normal``
  * ``Generator.standard_t``
  * ``Generator.triangular``
  * ``Generator.uniform``
  * ``Generator.wald``
  * ``Generator.weibull``
  * ``Generator.zipf``

* The ``nbytes`` property on NumPy ``ndarray`` types is implemented.
* Nesting of nested-array types is now supported.
* ``datetime`` and ``timedelta`` types can be cast to ``int``.
* ``F``-order iteration is supported in ``ufunc`` generation for increased
  performance when using combinations of predominantly ``F``-order arrays.
* The following functions are also now supported:

  * ``np.argpartition``
  * ``np.isclose``
  * ``np.nan_to_num``
  * ``np.new_axis``
  * ``np.union1d``

Highlights of core changes:

* A large amount of refactoring has taken place to convert many of Numba's
  internal implementations, of both Python and NumPy functions, from the
  low-level extension API to the high-level extension API (``numba.extending``).
* The ``__repr__`` method is supported for Numba types.
* The default ``target`` for applicable functions in the extension API
  (``numba.extending``) is now ``"generic"``. This means that ``@overload*`` and
  ``@intrinsic`` functions will by default be accepted by both the CPU and CUDA
  targets.
* The use of ``__getitem__`` on Numba types is now supported in compiled code.
  i.e. ``types.float64[:, ::1]`` is now compilable.

Performance:

* The performance of ``str.find()`` and ``str.rfind()`` has been improved.
* Unicode support for ``__getitem__`` now avoids allocation and returns a view.
* The ``numba.typed.Dict`` dictionary now accepts an ``n_keys`` option to enable
  allocating the dictionary instance to a predetermined initial size (useful to
  avoid resizes!).
* The Numba Run-time (NRT) has been improved in terms of performance and safety:

  * The NRT internal statistics counters are now off by default (removes atomic
    lock contentions).
  * Debug cache line filling is off by default.
  * The NRT is only compiled once a compilation starts opposed to at function
    decoration time, this improves import speed.
  * The NRT allocation calls are all made through a "checked" layer by default.

CUDA:

* New NVIDIA hardware and software compatibility / support:

  * Toolkits: CUDA 11.8 and 12, with Minor Version Compatibility for 11.x.
  * Packaging: NVIDIA-packaged CUDA toolkit conda packages.
  * Hardware: Hopper, Ada Lovelace, and AGX Orin.

* ``float16`` support:

  * Arithmetic operations are now fully supported.
  * A new method, ``is_fp16_supported()``, and device property,
    ``supports_float16``, for checking the availability of ``float16`` support.

* Functionality:

  * The high-level extension API is now fully-supported in the CUDA target.
  * Eager compilation of multiple signatures, multiple outputs from generalized
    ufuncs, and specifying the return type of ufuncs are now supported.
  * A limited set of NumPy ufuncs (trigonometric functions) can now be called
    inside kernels.

* Lineinfo quality improvement: enabling lineinfo no longer results in any
  changes to generated code.

Deprecations:

* The ``numba.pycc`` module and everything in it is now pending deprecation.
* The long awaited full deprecation of ``object mode`` `fall-back` is
  underway. This change means ``@jit`` with no keyword arguments will eventually
  alias ``@njit``.
* The ``@generated_jit`` decorator is deprecated as the Numba extension API
  provides a better supported superset of the same functionality, particularly
  through ``@numba.extending.overload``.

Version support/dependency changes:

* The ``setuptools`` package is now an optional run-time dependency opposed to a
  required run-time dependency.
* The TBB threading-layer now requires version 2021.6 or later.
* LLVM 14 is now supported on all platforms via ``llvmlite``.

Pull-Requests:

* PR `#5113 <https://github.com/numba/numba/pull/5113>`_: Fix error handling in the Interval extending example (`esc <https://github.com/esc>`_ `eric-wieser <https://github.com/eric-wieser>`_)
* PR `#5544 <https://github.com/numba/numba/pull/5544>`_: Add support for np.union1d (`shangbol <https://github.com/shangbol>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#7009 <https://github.com/numba/numba/pull/7009>`_: Add writable args (`dmbelov <https://github.com/dmbelov>`_)
* PR `#7067 <https://github.com/numba/numba/pull/7067>`_: Implement np.isclose (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#7255 <https://github.com/numba/numba/pull/7255>`_: CUDA: Support CUDA Toolkit conda packages from NVIDIA (`gmarkall <https://github.com/gmarkall>`_)
* PR `#7622 <https://github.com/numba/numba/pull/7622>`_: Support fortran loop ordering for ufunc generation (`sklam <https://github.com/sklam>`_)
* PR `#7733 <https://github.com/numba/numba/pull/7733>`_: fix for /tmp/tmp access issues (`ChiCheng45 <https://github.com/ChiCheng45>`_)
* PR `#7884 <https://github.com/numba/numba/pull/7884>`_: Implement getattr builtin. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7885 <https://github.com/numba/numba/pull/7885>`_: Adds CUDA FP16 arithmetic operators (`testhound <https://github.com/testhound>`_)
* PR `#7920 <https://github.com/numba/numba/pull/7920>`_: Drop pre-3.7 code path (CPU only) (`sklam <https://github.com/sklam>`_)
* PR `#8001 <https://github.com/numba/numba/pull/8001>`_: CUDA fp16 math functions (`testhound <https://github.com/testhound>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8010 <https://github.com/numba/numba/pull/8010>`_: Add support for fp16 comparison native operators (`testhound <https://github.com/testhound>`_)
* PR `#8024 <https://github.com/numba/numba/pull/8024>`_: Allow converting NumPy datetimes to int (`apmasell <https://github.com/apmasell>`_)
* PR `#8038 <https://github.com/numba/numba/pull/8038>`_: Support for Numpy BitGenerators PR#2: Standard Distributions support (`kc611 <https://github.com/kc611>`_)
* PR `#8040 <https://github.com/numba/numba/pull/8040>`_: Support for Numpy BitGenerators PR#3: Advanced Distributions Support. (`kc611 <https://github.com/kc611>`_)
* PR `#8041 <https://github.com/numba/numba/pull/8041>`_: Support for Numpy BitGenerators PR#4: Generator().integers() Support. (`kc611 <https://github.com/kc611>`_)
* PR `#8042 <https://github.com/numba/numba/pull/8042>`_: Support for NumPy BitGenerators PR#5: Generator Shuffling Methods. (`kc611 <https://github.com/kc611>`_)
* PR `#8061 <https://github.com/numba/numba/pull/8061>`_: Migrate random ``glue_lowering`` to ``overload`` where easy (`apmasell <https://github.com/apmasell>`_)
* PR `#8106 <https://github.com/numba/numba/pull/8106>`_: Remove injection of atomic JIT functions into NRT memsys. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8120 <https://github.com/numba/numba/pull/8120>`_: Support nesting of nested array types (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8134 <https://github.com/numba/numba/pull/8134>`_: Support non-constant exception values in JIT (`guilhermeleobas <https://github.com/guilhermeleobas>`_ `sklam <https://github.com/sklam>`_)
* PR `#8147 <https://github.com/numba/numba/pull/8147>`_: Adds size variable at runtime for arrays that cannot be inferred  (`njriasan <https://github.com/njriasan>`_)
* PR `#8154 <https://github.com/numba/numba/pull/8154>`_: Testhound/native cast 8138 (`testhound <https://github.com/testhound>`_)
* PR `#8158 <https://github.com/numba/numba/pull/8158>`_: adding -pthread for linux-ppc64le in setup.py (`esc <https://github.com/esc>`_)
* PR `#8164 <https://github.com/numba/numba/pull/8164>`_: remove myself from automatic reviewer assignment (`esc <https://github.com/esc>`_)
* PR `#8167 <https://github.com/numba/numba/pull/8167>`_: CUDA: Facilitate and document passing arrays / pointers to foreign functions (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8180 <https://github.com/numba/numba/pull/8180>`_: CUDA: Initial support for Minor Version Compatibility (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8183 <https://github.com/numba/numba/pull/8183>`_: Add ``n_keys`` option to Dict.empty() (`stefanfed <https://github.com/stefanfed>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8198 <https://github.com/numba/numba/pull/8198>`_: Update the release template to include updating the version table. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8200 <https://github.com/numba/numba/pull/8200>`_: Make the NRT use the "unsafe" allocation API by default. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8201 <https://github.com/numba/numba/pull/8201>`_: Bump llvmlite dependency to 0.40.dev0 for Numba 0.57.0dev0 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8207 <https://github.com/numba/numba/pull/8207>`_: development tag should be in monofont (`esc <https://github.com/esc>`_)
* PR `#8212 <https://github.com/numba/numba/pull/8212>`_: release checklist: include a note to ping @RC_testers on discourse (`esc <https://github.com/esc>`_)
* PR `#8216 <https://github.com/numba/numba/pull/8216>`_: chore: Set permissions for GitHub actions (`naveensrinivasan <https://github.com/naveensrinivasan>`_)
* PR `#8217 <https://github.com/numba/numba/pull/8217>`_: Fix syntax in docs (`jorgepiloto <https://github.com/jorgepiloto>`_)
* PR `#8220 <https://github.com/numba/numba/pull/8220>`_: Added the interval example as doctest (`kc611 <https://github.com/kc611>`_)
* PR `#8221 <https://github.com/numba/numba/pull/8221>`_: CUDA stubs docstring: Replace illegal escape sequence (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8228 <https://github.com/numba/numba/pull/8228>`_: Fix typo in @vectorize docstring and a NumPy spelling. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8229 <https://github.com/numba/numba/pull/8229>`_: Remove ``mk_unique_var`` in ``inline_closurecall.py`` (`sklam <https://github.com/sklam>`_)
* PR `#8234 <https://github.com/numba/numba/pull/8234>`_: Replace @overload_glue by @overload for 20 NumPy functions (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8235 <https://github.com/numba/numba/pull/8235>`_: Make the NRT stats counters optional. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8238 <https://github.com/numba/numba/pull/8238>`_: Advanced Indexing Support #1 (`kc611 <https://github.com/kc611>`_)
* PR `#8240 <https://github.com/numba/numba/pull/8240>`_: Add get_shared_mem_per_block method to Dispatcher  (`testhound <https://github.com/testhound>`_)
* PR `#8241 <https://github.com/numba/numba/pull/8241>`_: Reorder typeof checks to avoid infinite loops on StructrefProxy  __hash__ (`DannyWeitekamp <https://github.com/DannyWeitekamp>`_)
* PR `#8243 <https://github.com/numba/numba/pull/8243>`_: Add a note to ``reference/numpysupported.rst`` ()
* PR `#8245 <https://github.com/numba/numba/pull/8245>`_: Fix links in ``CONTRIBUTING.md`` ()
* PR `#8247 <https://github.com/numba/numba/pull/8247>`_: Fix issue 8127 (`bszollosinagy <https://github.com/bszollosinagy>`_)
* PR `#8250 <https://github.com/numba/numba/pull/8250>`_: Fix issue 8161 (`bszollosinagy <https://github.com/bszollosinagy>`_)
* PR `#8253 <https://github.com/numba/numba/pull/8253>`_: CUDA: Verify NVVM IR prior to compilation (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8255 <https://github.com/numba/numba/pull/8255>`_: CUDA: Make numba.cuda.tests.doc_examples.ffi a module to fix #8252 (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8256 <https://github.com/numba/numba/pull/8256>`_: Migrate linear algebra functions from glue_lowering (`apmasell <https://github.com/apmasell>`_)
* PR `#8258 <https://github.com/numba/numba/pull/8258>`_: refactor np.where to use overload (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8259 <https://github.com/numba/numba/pull/8259>`_: Add ``np.broadcast_to(scalar_array, ())`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8264 <https://github.com/numba/numba/pull/8264>`_: remove ``mk_unique_var`` from ``parfor_lowering_utils.py`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8265 <https://github.com/numba/numba/pull/8265>`_: Remove ``mk_unique_var`` from ``array_analysis.py`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8266 <https://github.com/numba/numba/pull/8266>`_: Remove ``mk_unique_var`` in ``untyped_passes.py`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8267 <https://github.com/numba/numba/pull/8267>`_: Fix segfault for invalid axes in np.split (`aseyboldt <https://github.com/aseyboldt>`_)
* PR `#8271 <https://github.com/numba/numba/pull/8271>`_: Implement some CUDA intrinsics with ``@overload``, ``@overload_attribute``, and ``@intrinsic`` (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8274 <https://github.com/numba/numba/pull/8274>`_: Update version support table doc for 0.56. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8275 <https://github.com/numba/numba/pull/8275>`_: Update CHANGE_LOG for 0.56.0 final (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8283 <https://github.com/numba/numba/pull/8283>`_: Clean up / remove support for old NumPy versions (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8287 <https://github.com/numba/numba/pull/8287>`_: Drop CUDA 10.2 (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8289 <https://github.com/numba/numba/pull/8289>`_: Revert #8265. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8290 <https://github.com/numba/numba/pull/8290>`_: CUDA: Replace use of deprecated NVVM IR features, questionable constructs (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8292 <https://github.com/numba/numba/pull/8292>`_: update checklist (`esc <https://github.com/esc>`_)
* PR `#8294 <https://github.com/numba/numba/pull/8294>`_: CUDA: Add trig ufunc support (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8295 <https://github.com/numba/numba/pull/8295>`_: Add get_const_mem_size method to Dispatcher (`testhound <https://github.com/testhound>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8297 <https://github.com/numba/numba/pull/8297>`_: Add __name__ attribute to CUDAUFuncDispatcher and test case (`testhound <https://github.com/testhound>`_)
* PR `#8299 <https://github.com/numba/numba/pull/8299>`_: Fix build for mingw toolchain (`Biswa96 <https://github.com/Biswa96>`_)
* PR `#8302 <https://github.com/numba/numba/pull/8302>`_: CUDA: Revert numba_nvvm intrinsic name workaround (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8308 <https://github.com/numba/numba/pull/8308>`_: CUDA: Support for multiple signatures (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8315 <https://github.com/numba/numba/pull/8315>`_: Add get_local_mem_per_thread method to Dispatcher (`testhound <https://github.com/testhound>`_)
* PR `#8319 <https://github.com/numba/numba/pull/8319>`_: Bump minimum supported Python version to 3.8 (`esc <https://github.com/esc>`_ `stuartarchibald <https://github.com/stuartarchibald>`_ `jamesobutler <https://github.com/jamesobutler>`_)
* PR `#8320 <https://github.com/numba/numba/pull/8320>`_: Add __name__ support for GUFuncs (`testhound <https://github.com/testhound>`_)
* PR `#8321 <https://github.com/numba/numba/pull/8321>`_: Fix literal_unroll pass erroneously exiting on non-conformant loop. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8325 <https://github.com/numba/numba/pull/8325>`_: Remove use of mk_unique_var in stencil.py (`bszollosinagy <https://github.com/bszollosinagy>`_)
* PR `#8326 <https://github.com/numba/numba/pull/8326>`_: Remove ``mk_unique_var`` from ``parfor_lowering.py`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8331 <https://github.com/numba/numba/pull/8331>`_: Extend docs with info on how to call C functions from Numba (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8334 <https://github.com/numba/numba/pull/8334>`_: Add dict(\*iterable) constructor (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8335 <https://github.com/numba/numba/pull/8335>`_: Remove deprecated pycc script and related source. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8336 <https://github.com/numba/numba/pull/8336>`_: Fix typos of "Generalized" in GUFunc-related code (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8338 <https://github.com/numba/numba/pull/8338>`_: Calculate reductions before fusion so that use of reduction vars can stop fusion. (`DrTodd13 <https://github.com/DrTodd13>`_)
* PR `#8339 <https://github.com/numba/numba/pull/8339>`_: Fix #8291 parfor leak of redtoset variable (`sklam <https://github.com/sklam>`_)
* PR `#8341 <https://github.com/numba/numba/pull/8341>`_: CUDA: Support multiple outputs for Generalized Ufuncs (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8343 <https://github.com/numba/numba/pull/8343>`_: Eliminate references to type annotation in compile_ptx (`testhound <https://github.com/testhound>`_)
* PR `#8348 <https://github.com/numba/numba/pull/8348>`_: Add get_max_threads_per_block method to Dispatcher (`testhound <https://github.com/testhound>`_)
* PR `#8354 <https://github.com/numba/numba/pull/8354>`_: pin setuptools to < 65 and switch from mamba to conda on RTD (`esc <https://github.com/esc>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8357 <https://github.com/numba/numba/pull/8357>`_: Clean up the buildscripts directory. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8359 <https://github.com/numba/numba/pull/8359>`_: adding warnings about cache behaviour (`luk-f-a <https://github.com/luk-f-a>`_)
* PR `#8368 <https://github.com/numba/numba/pull/8368>`_: Remove ``glue_lowering`` in random math that requires IR (`apmasell <https://github.com/apmasell>`_)
* PR `#8376 <https://github.com/numba/numba/pull/8376>`_: Fix issue 8370 (`bszollosinagy <https://github.com/bszollosinagy>`_)
* PR `#8387 <https://github.com/numba/numba/pull/8387>`_: Add support for compute capability in IR Lowering (`testhound <https://github.com/testhound>`_)
* PR `#8388 <https://github.com/numba/numba/pull/8388>`_: Remove more references to the pycc binary. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8389 <https://github.com/numba/numba/pull/8389>`_: Make C++ extensions compile with correct compiler (`apmasell <https://github.com/apmasell>`_)
* PR `#8390 <https://github.com/numba/numba/pull/8390>`_: Use NumPy logic for lessthan in sort to move NaNs to the back. (`sklam <https://github.com/sklam>`_)
* PR `#8401 <https://github.com/numba/numba/pull/8401>`_: Remove Cuda toolkit version check (`testhound <https://github.com/testhound>`_)
* PR `#8415 <https://github.com/numba/numba/pull/8415>`_: Refactor ``numba.np.arraymath`` methods from lower_builtins to overloads (`kc611 <https://github.com/kc611>`_)
* PR `#8418 <https://github.com/numba/numba/pull/8418>`_: Fixes ravel failure on 1d arrays (#5229) (`cako <https://github.com/cako>`_)
* PR `#8421 <https://github.com/numba/numba/pull/8421>`_: Update release checklist: add a task to check dependency pinnings on subsequent releases (e.g. PATCH) (`esc <https://github.com/esc>`_)
* PR `#8422 <https://github.com/numba/numba/pull/8422>`_: Switch public CI builds to use gdb from conda packages. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8423 <https://github.com/numba/numba/pull/8423>`_: Remove public facing and CI references to 32 bit linux support. (`stuartarchibald <https://github.com/stuartarchibald>`_,
  in addition, we are grateful for the contribution of `jamesobutler <https://github.com/jamesobutler>`_ towards a similar goal in PR `#8319 <https://github.com/numba/numba/pull/8319>`_)
* PR `#8425 <https://github.com/numba/numba/pull/8425>`_: Post 0.56.2 cleanup (`esc <https://github.com/esc>`_)
* PR `#8427 <https://github.com/numba/numba/pull/8427>`_: Shorten the time to verify test discovery. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8429 <https://github.com/numba/numba/pull/8429>`_: changelog generator script (`esc <https://github.com/esc>`_)
* PR `#8431 <https://github.com/numba/numba/pull/8431>`_: Replace ``@overload_glue`` by ``@overload`` for ``np.linspace`` and ``np.take`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8432 <https://github.com/numba/numba/pull/8432>`_: Refactor carray/farray to use @overload (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8435 <https://github.com/numba/numba/pull/8435>`_: Migrate ``np.atleast_?`` functions from ``glue_lowering`` to ``overload`` (`apmasell <https://github.com/apmasell>`_)
* PR `#8438 <https://github.com/numba/numba/pull/8438>`_: Make the initialisation of the NRT more lazy for the njit decorator. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8439 <https://github.com/numba/numba/pull/8439>`_: Update the contributing docs to include a policy on formatting changes. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8440 <https://github.com/numba/numba/pull/8440>`_: [DOC]: Replaces icc_rt with intel-cmplr-lib-rt (`oleksandr-pavlyk <https://github.com/oleksandr-pavlyk>`_)
* PR `#8442 <https://github.com/numba/numba/pull/8442>`_: Implement hasattr(), str() and repr(). (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8446 <https://github.com/numba/numba/pull/8446>`_: add version info in ImportError's (`raybellwaves <https://github.com/raybellwaves>`_)
* PR `#8450 <https://github.com/numba/numba/pull/8450>`_: remove GitHub username from changelog generation script (`esc <https://github.com/esc>`_)
* PR `#8467 <https://github.com/numba/numba/pull/8467>`_: Convert implementations using generated_jit to overload (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8468 <https://github.com/numba/numba/pull/8468>`_: Reference test suite in installation documentation (`apmasell <https://github.com/apmasell>`_)
* PR `#8469 <https://github.com/numba/numba/pull/8469>`_: Correctly handle optional types in parfors lowering (`apmasell <https://github.com/apmasell>`_)
* PR `#8473 <https://github.com/numba/numba/pull/8473>`_: change the include style in _pymodule.h and remove unused or duplicate headers in two header files ()
* PR `#8476 <https://github.com/numba/numba/pull/8476>`_: Make setuptools optional at runtime. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8490 <https://github.com/numba/numba/pull/8490>`_: Restore installing SciPy from defaults instead of conda-forge on public CI (`esc <https://github.com/esc>`_)
* PR `#8494 <https://github.com/numba/numba/pull/8494>`_: Remove ``context.compile_internal`` where easy on ``numba/cpython/cmathimpl.py`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8495 <https://github.com/numba/numba/pull/8495>`_: Removes context.compile_internal where easy on ``numba/cpython/listobj.py`` (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8496 <https://github.com/numba/numba/pull/8496>`_: Rewrite most of the set API to use overloads (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8499 <https://github.com/numba/numba/pull/8499>`_: Deprecate numba.generated_jit (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8508 <https://github.com/numba/numba/pull/8508>`_: This updates the release checklists to capture some more checks. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8513 <https://github.com/numba/numba/pull/8513>`_: Added support for numpy.newaxis (`kc611 <https://github.com/kc611>`_)
* PR `#8517 <https://github.com/numba/numba/pull/8517>`_: make some typedlist C-APIs public ()
* PR `#8518 <https://github.com/numba/numba/pull/8518>`_: Adjust stencil tests to use hardcoded python source opposed to AST. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8520 <https://github.com/numba/numba/pull/8520>`_: Added noncentral-chisquared, noncentral-f and logseries distributions (`kc611 <https://github.com/kc611>`_)
* PR `#8522 <https://github.com/numba/numba/pull/8522>`_: Import jitclass from numba.experimental in jitclass documentation (`armgabrielyan <https://github.com/armgabrielyan>`_)
* PR `#8524 <https://github.com/numba/numba/pull/8524>`_: Fix grammar in stencil.rst (`armgabrielyan <https://github.com/armgabrielyan>`_)
* PR `#8525 <https://github.com/numba/numba/pull/8525>`_: Making CUDA specific datamodel manager (`sklam <https://github.com/sklam>`_)
* PR `#8526 <https://github.com/numba/numba/pull/8526>`_: Fix broken url (`Nimrod0901 <https://github.com/Nimrod0901>`_)
* PR `#8527 <https://github.com/numba/numba/pull/8527>`_: Fix grammar in troubleshoot.rst (`armgabrielyan <https://github.com/armgabrielyan>`_)
* PR `#8532 <https://github.com/numba/numba/pull/8532>`_: Vary NumPy version on gpuCI (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8535 <https://github.com/numba/numba/pull/8535>`_: LLVM14 (`apmasell <https://github.com/apmasell>`_)
* PR `#8536 <https://github.com/numba/numba/pull/8536>`_: Fix fusion bug. (`DrTodd13 <https://github.com/DrTodd13>`_)
* PR `#8539 <https://github.com/numba/numba/pull/8539>`_: Fix #8534, np.broadcast_to should update array size attr. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8541 <https://github.com/numba/numba/pull/8541>`_: Remove restoration of "free" channel in Azure CI windows builds. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8542 <https://github.com/numba/numba/pull/8542>`_: CUDA: Make arg optional for Stream.add_callback() (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8544 <https://github.com/numba/numba/pull/8544>`_: Remove reliance on npy_<impl> ufunc loops. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8545 <https://github.com/numba/numba/pull/8545>`_: Py3.11 basic support (`esc <https://github.com/esc>`_ `sklam <https://github.com/sklam>`_)
* PR `#8547 <https://github.com/numba/numba/pull/8547>`_: [Unicode] Add more string view usages for unicode operations ()
* PR `#8549 <https://github.com/numba/numba/pull/8549>`_: Fix rstcheck in Azure CI builds, update sphinx dep and docs to match (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8550 <https://github.com/numba/numba/pull/8550>`_: Changes how tests are split between test instances (`apmasell <https://github.com/apmasell>`_)
* PR `#8554 <https://github.com/numba/numba/pull/8554>`_: Make target for ``@overload`` have 'generic' as default. (`stuartarchibald <https://github.com/stuartarchibald>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8557 <https://github.com/numba/numba/pull/8557>`_: [Unicode] support startswith with args, start and end. ()
* PR `#8566 <https://github.com/numba/numba/pull/8566>`_: Update workqueue abort message on concurrent access. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8572 <https://github.com/numba/numba/pull/8572>`_: CUDA: Reduce memory pressure from local memory tests (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8579 <https://github.com/numba/numba/pull/8579>`_: CUDA: Add CUDA 11.8 / Hopper support and required fixes (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8580 <https://github.com/numba/numba/pull/8580>`_: adding note about doing a wheel test build prior to tagging (`esc <https://github.com/esc>`_)
* PR `#8583 <https://github.com/numba/numba/pull/8583>`_: Skip tests that contribute to M1 RuntimeDyLd Assertion error  (`sklam <https://github.com/sklam>`_)
* PR `#8587 <https://github.com/numba/numba/pull/8587>`_: Remove unused refcount removal code, clean ``core/cpu.py`` module. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8588 <https://github.com/numba/numba/pull/8588>`_: Remove lowering extension hooks, replace with pass infrastructure. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8590 <https://github.com/numba/numba/pull/8590>`_: Py3.11 support continues (`sklam <https://github.com/sklam>`_)
* PR `#8592 <https://github.com/numba/numba/pull/8592>`_: fix failure of test_cache_invalidate due to read-only install (`tpwrules <https://github.com/tpwrules>`_)
* PR `#8593 <https://github.com/numba/numba/pull/8593>`_: Adjusted ULP precesion for noncentral distribution test (`kc611 <https://github.com/kc611>`_)
* PR `#8594 <https://github.com/numba/numba/pull/8594>`_: Fix various CUDA lineinfo issues (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8597 <https://github.com/numba/numba/pull/8597>`_: Prevent use of NumPy's MaskedArray. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8598 <https://github.com/numba/numba/pull/8598>`_: Setup Azure CI to test py3.11 (`sklam <https://github.com/sklam>`_)
* PR `#8600 <https://github.com/numba/numba/pull/8600>`_: Chrome trace timestamp should be in microseconds not seconds. (`sklam <https://github.com/sklam>`_)
* PR `#8602 <https://github.com/numba/numba/pull/8602>`_: Throw error for unsupported dunder methods (`apmasell <https://github.com/apmasell>`_)
* PR `#8605 <https://github.com/numba/numba/pull/8605>`_: Support for CUDA fp16 math functions (part 1) (`testhound <https://github.com/testhound>`_)
* PR `#8606 <https://github.com/numba/numba/pull/8606>`_: [Doc] Make the RewriteArrayExprs doc more precise ()
* PR `#8619 <https://github.com/numba/numba/pull/8619>`_: Added flat iteration logic for random distributions (`kc611 <https://github.com/kc611>`_)
* PR `#8623 <https://github.com/numba/numba/pull/8623>`_: Adds support for np.nan_to_num (`thomasjpfan <https://github.com/thomasjpfan>`_)
* PR `#8624 <https://github.com/numba/numba/pull/8624>`_: DOC: Add guvectorize scalar return example (`Matt711 <https://github.com/Matt711>`_)
* PR `#8625 <https://github.com/numba/numba/pull/8625>`_: Refactor ``test_ufuncs`` (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8626 <https://github.com/numba/numba/pull/8626>`_: [unicode-PERF]: use optmized BM algorithm to replace the brute-force finder (`dlee992 <https://github.com/dlee992>`_)
* PR `#8630 <https://github.com/numba/numba/pull/8630>`_: Fix #8628: Don't test math.trunc with non-float64 NumPy scalars (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8634 <https://github.com/numba/numba/pull/8634>`_: Add new method is_fp16_supported (`testhound <https://github.com/testhound>`_)
* PR `#8636 <https://github.com/numba/numba/pull/8636>`_: CUDA: Skip ``test_ptds`` on Windows (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8639 <https://github.com/numba/numba/pull/8639>`_: Python 3.11 - fix majority of remaining test failures. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8644 <https://github.com/numba/numba/pull/8644>`_: Fix bare reraise support (`sklam <https://github.com/sklam>`_)
* PR `#8649 <https://github.com/numba/numba/pull/8649>`_: Remove ``numba.core.overload_glue`` module. (`apmasell <https://github.com/apmasell>`_)
* PR `#8659 <https://github.com/numba/numba/pull/8659>`_: Preserve module name of jitted class (`neilflood <https://github.com/neilflood>`_)
* PR `#8661 <https://github.com/numba/numba/pull/8661>`_: Make external compiler discovery lazy in the test suite. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8662 <https://github.com/numba/numba/pull/8662>`_: Add support for ``.nbytes`` accessor for numpy arrays (`alanhdu <https://github.com/alanhdu>`_)
* PR `#8666 <https://github.com/numba/numba/pull/8666>`_: Updates for Python 3.8 baseline/Python 3.11 migration (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8673 <https://github.com/numba/numba/pull/8673>`_: Enable the CUDA simulator tests on Windows builds in Azure CI. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8675 <https://github.com/numba/numba/pull/8675>`_: Make ``always_run`` test decorator a tag and improve shard tests. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8677 <https://github.com/numba/numba/pull/8677>`_: Add support for min and max on boolean types. (`DrTodd13 <https://github.com/DrTodd13>`_)
* PR `#8680 <https://github.com/numba/numba/pull/8680>`_: Adjust flake8 config to be compatible with flake8=6.0.0 (`thomasjpfan <https://github.com/thomasjpfan>`_)
* PR `#8685 <https://github.com/numba/numba/pull/8685>`_: Implement ``__repr__`` for numba types (`luk-f-a <https://github.com/luk-f-a>`_)
* PR `#8691 <https://github.com/numba/numba/pull/8691>`_: NumPy 1.24 (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8697 <https://github.com/numba/numba/pull/8697>`_: Close stale issues after 7 days (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8701 <https://github.com/numba/numba/pull/8701>`_: Relaxed ULP testing precision for NumPy Generator tests across all systems (`kc611 <https://github.com/kc611>`_)
* PR `#8702 <https://github.com/numba/numba/pull/8702>`_: Supply concrete timeline for objmode fallback deprecation/removal. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8706 <https://github.com/numba/numba/pull/8706>`_: Fix doctest for ``@vectorize`` (`sklam <https://github.com/sklam>`_)
* PR `#8711 <https://github.com/numba/numba/pull/8711>`_: Python 3.11 tracing support (continuation of #8670). (`AndrewVallette <https://github.com/AndrewVallette>`_ `sklam <https://github.com/sklam>`_)
* PR `#8716 <https://github.com/numba/numba/pull/8716>`_: CI: Use ``set -e`` in "Before Install" step and fix install (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8720 <https://github.com/numba/numba/pull/8720>`_: Enable coverage for subprocess testing (`sklam <https://github.com/sklam>`_)
* PR `#8723 <https://github.com/numba/numba/pull/8723>`_: Check for void return type in ``cuda.compile_ptx`` (`brandonwillard <https://github.com/brandonwillard>`_)
* PR `#8726 <https://github.com/numba/numba/pull/8726>`_: Make Numba dependency check run ahead of Numba internal imports. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8728 <https://github.com/numba/numba/pull/8728>`_: Fix flake8 checks since upgrade to flake8=6.x (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8729 <https://github.com/numba/numba/pull/8729>`_: Run flake8 CI step in multiple processes. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8732 <https://github.com/numba/numba/pull/8732>`_: Add numpy argpartition function support ()
* PR `#8735 <https://github.com/numba/numba/pull/8735>`_: Update bot to close PRs waiting on authors for more than 3 months (`guilhermeleobas <https://github.com/guilhermeleobas>`_)
* PR `#8736 <https://github.com/numba/numba/pull/8736>`_: Implement np.lib.stride_tricks.sliding_window_view ()
* PR `#8744 <https://github.com/numba/numba/pull/8744>`_: Update CtypesLinker::add_cu error message to include fp16 usage (`testhound <https://github.com/testhound>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8746 <https://github.com/numba/numba/pull/8746>`_: Fix failing test_dispatcher test case (`testhound <https://github.com/testhound>`_)
* PR `#8748 <https://github.com/numba/numba/pull/8748>`_: Suppress known test failures for py3.11 (`sklam <https://github.com/sklam>`_)
* PR `#8751 <https://github.com/numba/numba/pull/8751>`_: Recycle test runners more aggressively (`apmasell <https://github.com/apmasell>`_)
* PR `#8752 <https://github.com/numba/numba/pull/8752>`_: Flake8 fixes for py311 branch (`esc <https://github.com/esc>`_ `sklam <https://github.com/sklam>`_)
* PR `#8760 <https://github.com/numba/numba/pull/8760>`_: Bump llvmlite PR in py3.11 branch testing (`sklam <https://github.com/sklam>`_)
* PR `#8764 <https://github.com/numba/numba/pull/8764>`_: CUDA tidy-up: remove some unneeded methods (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8765 <https://github.com/numba/numba/pull/8765>`_: BLD: remove distutils (`fangchenli <https://github.com/fangchenli>`_)
* PR `#8766 <https://github.com/numba/numba/pull/8766>`_: Stale bot: Use ``abandoned - stale`` label for closed PRs (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8771 <https://github.com/numba/numba/pull/8771>`_: Update vendored Versioneer from 0.14 to 0.28 (`oscargus <https://github.com/oscargus>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8775 <https://github.com/numba/numba/pull/8775>`_: Revert PR#8751 for buildfarm stability (`sklam <https://github.com/sklam>`_)
* PR `#8780 <https://github.com/numba/numba/pull/8780>`_: Improved documentation for Atomic CAS (`MiloniAtal <https://github.com/MiloniAtal>`_)
* PR `#8781 <https://github.com/numba/numba/pull/8781>`_: Ensure gc.collect() is called before checking refcount in tests. (`sklam <https://github.com/sklam>`_)
* PR `#8782 <https://github.com/numba/numba/pull/8782>`_: Changed wording of the escape error (`MiloniAtal <https://github.com/MiloniAtal>`_)
* PR `#8786 <https://github.com/numba/numba/pull/8786>`_: Upgrade stale GitHub action (`apmasell <https://github.com/apmasell>`_)
* PR `#8788 <https://github.com/numba/numba/pull/8788>`_: CUDA: Fix returned dtype of vectorized functions (Issue #8400) (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8790 <https://github.com/numba/numba/pull/8790>`_: CUDA compare and swap with index (`ianthomas23 <https://github.com/ianthomas23>`_)
* PR `#8795 <https://github.com/numba/numba/pull/8795>`_: Add pending-deprecation warnings for ``numba.pycc`` (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8802 <https://github.com/numba/numba/pull/8802>`_: Move the minimum supported NumPy version to 1.21 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8803 <https://github.com/numba/numba/pull/8803>`_: Attempted fix to #8789 by changing ``compile_ptx`` to accept a signature instead of argument tuple (`KyanCheung <https://github.com/KyanCheung>`_)
* PR `#8804 <https://github.com/numba/numba/pull/8804>`_: Split parfor pass into 3 parts (`DrTodd13 <https://github.com/DrTodd13>`_)
* PR `#8809 <https://github.com/numba/numba/pull/8809>`_: Update LLVM versions for 0.57 release (`apmasell <https://github.com/apmasell>`_)
* PR `#8810 <https://github.com/numba/numba/pull/8810>`_: Fix llvmlite dependency in meta.yaml (`sklam <https://github.com/sklam>`_)
* PR `#8816 <https://github.com/numba/numba/pull/8816>`_: Fix some buildfarm test failures (`sklam <https://github.com/sklam>`_)
* PR `#8819 <https://github.com/numba/numba/pull/8819>`_: Support "static" __getitem__ on Numba types in ``@njit`` code. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8822 <https://github.com/numba/numba/pull/8822>`_: Merge py3.11 branch to main (`esc <https://github.com/esc>`_ `AndrewVallette <https://github.com/AndrewVallette>`_ `stuartarchibald <https://github.com/stuartarchibald>`_ `sklam <https://github.com/sklam>`_)
* PR `#8826 <https://github.com/numba/numba/pull/8826>`_: CUDA CFFI test: conditionally require cffi module (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8831 <https://github.com/numba/numba/pull/8831>`_: Redo py3.11 sync branch with main (`sklam <https://github.com/sklam>`_)
* PR `#8833 <https://github.com/numba/numba/pull/8833>`_: Fix typeguard import hook location. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8836 <https://github.com/numba/numba/pull/8836>`_: Fix failing typeguard test. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8837 <https://github.com/numba/numba/pull/8837>`_: Update AzureCI matrix for Python 3.11/NumPy 1.21..1.24 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8839 <https://github.com/numba/numba/pull/8839>`_: Add Dynamic Shared Memory example. (`k1m190r <https://github.com/k1m190r>`_)
* PR `#8842 <https://github.com/numba/numba/pull/8842>`_: Fix buildscripts, setup.py, docs for setuptools becoming optional. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8843 <https://github.com/numba/numba/pull/8843>`_: Pin typeguard to 3.0.1 in AzureCI. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8848 <https://github.com/numba/numba/pull/8848>`_: added lifted loops to glossary term (`cherieliu <https://github.com/cherieliu>`_)
* PR `#8852 <https://github.com/numba/numba/pull/8852>`_: Disable SLP vectorisation due to miscompilations. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8855 <https://github.com/numba/numba/pull/8855>`_: DOC: ``pip`` into double backticks in installing.rst (`F3eQnxN3RriK <https://github.com/F3eQnxN3RriK>`_)
* PR `#8856 <https://github.com/numba/numba/pull/8856>`_: Update TBB to use >= 2021.6 by default.  (`kozlov-alexey <https://github.com/kozlov-alexey>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8858 <https://github.com/numba/numba/pull/8858>`_: Update deprecation notice for objmode fallback RE ``@jit`` use. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8864 <https://github.com/numba/numba/pull/8864>`_: Remove obsolete deprecation notices (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8866 <https://github.com/numba/numba/pull/8866>`_: Revise CUDA deprecation notices (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8869 <https://github.com/numba/numba/pull/8869>`_: Update CHANGE_LOG for 0.57.0rc1 (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_ `gmarkall <https://github.com/gmarkall>`_)
* PR `#8870 <https://github.com/numba/numba/pull/8870>`_: Fix opcode "spelling" change since Python 3.11 in CUDA debug test. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8879 <https://github.com/numba/numba/pull/8879>`_: Remove use of ``compile_isolated`` from generator tests. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8880 <https://github.com/numba/numba/pull/8880>`_: Fix missing dependency guard on pyyaml in ``test_azure_config``. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8881 <https://github.com/numba/numba/pull/8881>`_: Replace use of compile_isolated in test_obj_lifetime (`sklam <https://github.com/sklam>`_)
* PR `#8884 <https://github.com/numba/numba/pull/8884>`_: Pin llvmlite and NumPy on release branch (`sklam <https://github.com/sklam>`_)
* PR `#8887 <https://github.com/numba/numba/pull/8887>`_: Update PyPI supported version tags (`bryant1410 <https://github.com/bryant1410>`_)
* PR `#8896 <https://github.com/numba/numba/pull/8896>`_: Remove codecov install (now deleted from PyPI) (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8902 <https://github.com/numba/numba/pull/8902>`_: Enable CALL_FUNCTION_EX fix for py3.11 (`sklam <https://github.com/sklam>`_)
* PR `#8907 <https://github.com/numba/numba/pull/8907>`_: Work around issue #8898. Defer ``exp2`` (and ``log2``) calls to Numba internal symbols. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8909 <https://github.com/numba/numba/pull/8909>`_: Fix #8903. ``NumbaDeprecationWarning``s raised from ``@{gu,}vectorize``. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8929 <https://github.com/numba/numba/pull/8929>`_: Update CHANGE_LOG for 0.57.0 final. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8930 <https://github.com/numba/numba/pull/8930>`_: Fix year in change log (`jtilly <https://github.com/jtilly>`_)
* PR `#8932 <https://github.com/numba/numba/pull/8932>`_: Fix 0.57 release changelog (`sklam <https://github.com/sklam>`_)

Authors:

* `alanhdu <https://github.com/alanhdu>`_
* `AndrewVallette <https://github.com/AndrewVallette>`_
* `apmasell <https://github.com/apmasell>`_
* `armgabrielyan <https://github.com/armgabrielyan>`_
* `aseyboldt <https://github.com/aseyboldt>`_
* `Biswa96 <https://github.com/Biswa96>`_
* `brandonwillard <https://github.com/brandonwillard>`_
* `bryant1410 <https://github.com/bryant1410>`_
* `bszollosinagy <https://github.com/bszollosinagy>`_
* `cako <https://github.com/cako>`_
* `cherieliu <https://github.com/cherieliu>`_
* `ChiCheng45 <https://github.com/ChiCheng45>`_
* `DannyWeitekamp <https://github.com/DannyWeitekamp>`_
* `dlee992 <https://github.com/dlee992>`_
* `dmbelov <https://github.com/dmbelov>`_
* `DrTodd13 <https://github.com/DrTodd13>`_
* `eric-wieser <https://github.com/eric-wieser>`_
* `esc <https://github.com/esc>`_
* `F3eQnxN3RriK <https://github.com/F3eQnxN3RriK>`_
* `fangchenli <https://github.com/fangchenli>`_
* `gmarkall <https://github.com/gmarkall>`_
* `guilhermeleobas <https://github.com/guilhermeleobas>`_
* `ianthomas23 <https://github.com/ianthomas23>`_
* `jamesobutler <https://github.com/jamesobutler>`_
* `jorgepiloto <https://github.com/jorgepiloto>`_
* `jtilly <https://github.com/jtilly>`_
* `k1m190r <https://github.com/k1m190r>`_
* `kc611 <https://github.com/kc611>`_
* `kozlov-alexey <https://github.com/kozlov-alexey>`_
* `KyanCheung <https://github.com/KyanCheung>`_
* `luk-f-a <https://github.com/luk-f-a>`_
* `Matt711 <https://github.com/Matt711>`_
* `MiloniAtal <https://github.com/MiloniAtal>`_
* `naveensrinivasan <https://github.com/naveensrinivasan>`_
* `neilflood <https://github.com/neilflood>`_
* `Nimrod0901 <https://github.com/Nimrod0901>`_
* `njriasan <https://github.com/njriasan>`_
* `oleksandr-pavlyk <https://github.com/oleksandr-pavlyk>`_
* `oscargus <https://github.com/oscargus>`_
* `raybellwaves <https://github.com/raybellwaves>`_
* `shangbol <https://github.com/shangbol>`_
* `sklam <https://github.com/sklam>`_
* `stefanfed <https://github.com/stefanfed>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_
* `testhound <https://github.com/testhound>`_
* `thomasjpfan <https://github.com/thomasjpfan>`_
* `tpwrules <https://github.com/tpwrules>`_

Version 0.56.4 (3 November, 2022)
---------------------------------

This is a bugfix release to fix a regression in the CUDA target in relation to
the ``.view()`` method on CUDA device arrays that is present when using NumPy
version 1.23.0 or later.

Pull-Requests:

* PR `#8537 <https://github.com/numba/numba/pull/8537>`_: Make ol_compatible_view accessible on all targets (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8552 <https://github.com/numba/numba/pull/8552>`_: Update version support table for 0.56.4. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8553 <https://github.com/numba/numba/pull/8553>`_: Update CHANGE_LOG for 0.56.4 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8570 <https://github.com/numba/numba/pull/8570>`_: Release 0.56 branch: Fix overloads with ``target="generic"`` for CUDA (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8571 <https://github.com/numba/numba/pull/8571>`_: Additional update to CHANGE_LOG for 0.56.4 (`stuartarchibald <https://github.com/stuartarchibald>`_)

Authors:

* `gmarkall <https://github.com/gmarkall>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_

Version 0.56.3 (13 October, 2022)
---------------------------------

This is a bugfix release to remove the version restriction applied to the
``setuptools`` package and to fix a bug in the CUDA target in relation to
copying zero length device arrays to zero length host arrays.

Pull-Requests:

* PR `#8475 <https://github.com/numba/numba/pull/8475>`_: Remove setuptools version pin (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8482 <https://github.com/numba/numba/pull/8482>`_: Fix #8477: Allow copies with different strides for 0-length data (`gmarkall <https://github.com/gmarkall>`_)
* PR `#8486 <https://github.com/numba/numba/pull/8486>`_: Restrict the TBB development package to supported version in Azure. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8503 <https://github.com/numba/numba/pull/8503>`_: Update version support table for 0.56.3 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8504 <https://github.com/numba/numba/pull/8504>`_: Update CHANGE_LOG for 0.56.3 (`stuartarchibald <https://github.com/stuartarchibald>`_)

Authors:

* `gmarkall <https://github.com/gmarkall>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_

Version 0.56.2 (1 September, 2022)
----------------------------------

This is a bugfix release that supports NumPy 1.23 and fixes CUDA function
caching.

Pull-Requests:

* PR `#8239 <https://github.com/numba/numba/pull/8239>`_: Add decorator to run a test in a subprocess (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8276 <https://github.com/numba/numba/pull/8276>`_: Move Azure to use macos-11 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8310 <https://github.com/numba/numba/pull/8310>`_: CUDA: Fix Issue #8309 - atomics don't work on complex components (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8342 <https://github.com/numba/numba/pull/8342>`_: Upgrade to ubuntu-20.04 for azure pipeline CI (`jamesobutler <https://github.com/jamesobutler>`_)
* PR `#8356 <https://github.com/numba/numba/pull/8356>`_: Update setup.py, buildscripts, CI and docs to require setuptools<60 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8374 <https://github.com/numba/numba/pull/8374>`_: Don't pickle LLVM IR for CUDA code libraries (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8377 <https://github.com/numba/numba/pull/8377>`_: Add support for NumPy 1.23 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8384 <https://github.com/numba/numba/pull/8384>`_: Move strace() check into tests that actually need it (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8386 <https://github.com/numba/numba/pull/8386>`_: Fix the docs for numba.get_thread_id (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8407 <https://github.com/numba/numba/pull/8407>`_: Pin NumPy version to 1.18-1.24 (`Andre Masella <https://github.com/apmasell>`_)
* PR `#8411 <https://github.com/numba/numba/pull/8411>`_: update version support table for 0.56.1 (`esc <https://github.com/esc>`_)
* PR `#8412 <https://github.com/numba/numba/pull/8412>`_: Create changelog for 0.56.1 (`Andre Masella <https://github.com/apmasell>`_)
* PR `#8413 <https://github.com/numba/numba/pull/8413>`_: Fix Azure CI for NumPy 1.23 and use conda-forge scipy (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8414 <https://github.com/numba/numba/pull/8413>`_: Hotfix for 0.56.2 (`Siu Kwan Lam <https://github.com/sklam>`_)

Authors:

* `Andre Masella <https://github.com/apmasell>`_
* `esc <https://github.com/esc>`_
* `Graham Markall <https://github.com/gmarkall>`_
* `jamesobutler <https://github.com/jamesobutler>`_
* `Siu Kwan Lam <https://github.com/sklam>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_

Version 0.56.1 (NO RELEASE)
---------------------------

The release was skipped due to issues during the release process.

Version 0.56.0 (25 July, 2022)
------------------------------

This release continues to add new features, bug fixes and stability improvements
to Numba. Please note that this will be the last release that has support for
Python 3.7 as the next release series (Numba 0.57) will support Python 3.11!
Also note that, this will be the last release to support linux-32 packages
produced by the Numba team.

Python language support enhancements:

* Previously missing support for large, in-line dictionaries and internal calls
  to functions with large numbers of keyword arguments in Python 3.10 has been
  added.
* ``operator.mul`` now works for ``list`` s.
* Literal slices, e.g. ``slice(1, 10, 2)`` can be returned from ``nopython``
  mode functions.
* The ``len`` function now works on ``dict_keys``, ``dict_values`` and
  ``dict_items`` .
* Numba's ``set`` implementation now supports reference counted items e.g.
  strings.

Numba specific feature enhancements:

* The experimental ``jitclass`` feature gains support for a large number of
  ``builtin`` methods e.g. declaring ``__hash__`` or ``__getitem__`` for a
  ``jitclass`` type.
* It's now possible to use ``@vectorize`` on an already ``@jit`` family
  decorated function.
* Name mangling has been updated to emit compiled function names that exactly
  match the function name in Python. This means debuggers, like GDB, can be set
  to break directly on Python function names.
* A GDB "pretty printing" support module has been added, when loaded into GDB
  Numba's internal representations of Python/NumPy types are rendered inside GDB
  as they would be in Python.
* An experimental option is added to the ``@jit`` family decorators to entirely
  turn off LLVM's optimisation passes for a given function (see
  ``_dbg_optnone`` kwarg in the ``@jit`` decorator family).
* A new environment variable is added ``NUMBA_EXTEND_VARIABLE_LIFETIMES``, which
  if set will extend the lifetime of variables to the end of their basic block,
  this to permit a debugging experience in GDB similar to that found in compiled
  C/C++/Fortran code.

NumPy features/enhancements:

* Initial support for passing, using and returning ``numpy.random.Generator``
  instances has been added, this currently includes support for the ``random``
  distribution.
* The broadcasting functions ``np.broadcast_shapes`` and ``np.broadcast_arrays``
  are now supported.
* The ``min`` and ``max`` functions now work with ``np.timedelta64`` and
  ``np.datetime64`` types.
* Sorting multi-dimensional arrays along the last axis is now supported in
  ``np.sort()``.
* The ``np.clip`` function is updated to accept NumPy arrays for the ``a_min``
  and ``a_max`` arguments.
* The NumPy allocation routines (``np.empty`` , ``np.ones`` etc.) support shape
  arguments specified using members of ``enum.IntEnum`` s.
* The function ``np.random.noncentral_chisquare`` is now supported.
* The performance of functions ``np.full`` and ``np.ones`` has been improved.

Parallel Accelerator enhancements:

* The ``parallel=True`` functionality is enhanced through the addition of the
  functions ``numba.set_parallel_chunksize`` and
  ``numba.get_parallel_chunksize`` to permit a more fine grained scheduling of
  work defined in a parallel region. There is also support for adjusting the
  ``chunksize`` via a context manager.
* The ``ID`` of a thread is now defined to be predictable and within a known
  range, it is available through calling the function ``numba.get_thread_id``.
* The performance of ``@stencil`` s has been improved in both serial and
  parallel execution.

CUDA enhancements:

* New functionality:

  * Self-recursive device functions.
  * Vector type support (``float4``, ``int2``, etc.).
  * Shared / local arrays of extension types can now be created.
  * Support for linking CUDA C / C++ device functions into Python kernels.
  * PTX generation for Compute Capabilities 8.6 and 8.7 - e.g. RTX A series,
    GTX 3000 series.
  * Comparison operations for ``float16`` types.

* Performance improvements:

  * Context queries are no longer made during launch configuration.
  * Launch configurations are now LRU cached.
  * On-disk caching of CUDA kernels is now supported.

* Documentation: many new examples added.

Docs:

* Numba now has an official "mission statement".
* There's now a "version support table" in the documentation to act as an easy
  to use, single reference point, for looking up information about Numba
  releases and their required/supported dependencies.

General Enhancements:

* Numba imports more quickly in environments with large numbers of packages as
  it now uses ``importlib-metadata`` for querying other packages.
* Emission of chrome tracing output is now supported for the internal
  compilation event handling system.
* This release is tested and known to work when using the
  `Pyston <https://www.pyston.org/>`_ Python interpreter.

Pull-Requests:

* PR `#5209 <https://github.com/numba/numba/pull/5209>`_: Use importlib to load numba extensions (`Stepan Rakitin <https://github.com/svrakitin>`_ `Graham Markall <https://github.com/gmarkall>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#5877 <https://github.com/numba/numba/pull/5877>`_: Jitclass builtin methods (`Ethan Pronovost <https://github.com/EPronovost>`_ `Graham Markall <https://github.com/gmarkall>`_)
* PR `#6490 <https://github.com/numba/numba/pull/6490>`_: Stencil output allocated with np.empty now and new code to initialize the borders. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7005 <https://github.com/numba/numba/pull/7005>`_: Make `numpy.searchsorted` match NumPy when first argument is unsorted (`Brandon T. Willard <https://github.com/brandonwillard>`_)
* PR `#7363 <https://github.com/numba/numba/pull/7363>`_: Update cuda.local.array to clarify "simple constant expression" (e.g. no NumPy ints) (`Sterling Baird <https://github.com/sgbaird>`_)
* PR `#7364 <https://github.com/numba/numba/pull/7364>`_: Removes an instance of signed integer overflow undefined behaviour. (`Tobias Sargeant <https://github.com/folded>`_)
* PR `#7537 <https://github.com/numba/numba/pull/7537>`_: Add chrome tracing (`Hadia Ahmed <https://github.com/hadia206>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7556 <https://github.com/numba/numba/pull/7556>`_: Testhound/fp16 comparison (`Michael Collison <https://github.com/testhound>`_ `Graham Markall <https://github.com/gmarkall>`_)
* PR `#7586 <https://github.com/numba/numba/pull/7586>`_: Support for len on dict.keys, dict.values, and dict.items (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7617 <https://github.com/numba/numba/pull/7617>`_: Numba gdb-python extension for printing (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7619 <https://github.com/numba/numba/pull/7619>`_: CUDA: Fix linking with PTX when compiling lazily (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7621 <https://github.com/numba/numba/pull/7621>`_: Add support for linking CUDA C / C++ with `@cuda.jit` kernels (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7625 <https://github.com/numba/numba/pull/7625>`_: Combined parfor chunking and caching PRs. (`stuartarchibald <https://github.com/stuartarchibald>`_ `Todd A. Anderson <https://github.com/DrTodd13>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7651 <https://github.com/numba/numba/pull/7651>`_: DOC: pypi and conda-forge badges (`Ray Bell <https://github.com/raybellwaves>`_)
* PR `#7660 <https://github.com/numba/numba/pull/7660>`_: Add support for np.broadcast_arrays (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#7664 <https://github.com/numba/numba/pull/7664>`_: Flatten mangling dicts into a single dict (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7680 <https://github.com/numba/numba/pull/7680>`_: CUDA Docs: include example calling slow matmul (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7682 <https://github.com/numba/numba/pull/7682>`_: performance improvements to np.full and np.ones (`Rishi Kulkarni <https://github.com/rishi-kulkarni>`_)
* PR `#7684 <https://github.com/numba/numba/pull/7684>`_: DOC: remove incorrect warning in np.random reference (`Rishi Kulkarni <https://github.com/rishi-kulkarni>`_)
* PR `#7685 <https://github.com/numba/numba/pull/7685>`_: Don't convert setitems that have dimension mismatches to parfors. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7690 <https://github.com/numba/numba/pull/7690>`_: Implemented np.random.noncentral_chisquare for all size arguments (`Rishi Kulkarni <https://github.com/rishi-kulkarni>`_)
* PR `#7695 <https://github.com/numba/numba/pull/7695>`_: `IntEnumMember` support for  `np.empty`, `np.zeros`, and `np.ones` (`Benjamin Graham <https://github.com/benwilliamgraham>`_)
* PR `#7699 <https://github.com/numba/numba/pull/7699>`_: CUDA: Provide helpful error if the return type is missing for `declare_device` (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7700 <https://github.com/numba/numba/pull/7700>`_: Support for scalar arguments in Np.ascontiguousarray  (`Dhruv Patel <https://github.com/DhruvPatel01>`_)
* PR `#7703 <https://github.com/numba/numba/pull/7703>`_: Ignore unsupported types in `ShapeEquivSet._getnames()` (`Benjamin Graham <https://github.com/benwilliamgraham>`_)
* PR `#7704 <https://github.com/numba/numba/pull/7704>`_: Move the type annotation pass to post legalization. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7709 <https://github.com/numba/numba/pull/7709>`_: CUDA: Fixes missing type annotation pass following #7704 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7712 <https://github.com/numba/numba/pull/7712>`_: Fixing issue 7693 (`stuartarchibald <https://github.com/stuartarchibald>`_ `Graham Markall <https://github.com/gmarkall>`_ `luk-f-a <https://github.com/luk-f-a>`_)
* PR `#7714 <https://github.com/numba/numba/pull/7714>`_: Support for boxing SliceLiteral type (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7718 <https://github.com/numba/numba/pull/7718>`_: Bump llvmlite dependency to 0.39.0dev0 for Numba 0.56.0dev0 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7724 <https://github.com/numba/numba/pull/7724>`_: Update URLs in error messages to refer to RTD docs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7728 <https://github.com/numba/numba/pull/7728>`_: Document that AOT-compiled functions do not check arg types (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7729 <https://github.com/numba/numba/pull/7729>`_: Handle Omitted/OmittedArgDataModel in DI generation. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7732 <https://github.com/numba/numba/pull/7732>`_: update release checklist following 0.55.0 RC1 (`esc <https://github.com/esc>`_)
* PR `#7736 <https://github.com/numba/numba/pull/7736>`_: Update CHANGE_LOG for 0.55.0 final. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7740 <https://github.com/numba/numba/pull/7740>`_: CUDA Python 11.6 support (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7744 <https://github.com/numba/numba/pull/7744>`_: Fix issues with locating/parsing source during DebugInfo emission. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7745 <https://github.com/numba/numba/pull/7745>`_: Fix the release year for Numba 0.55 change log entry. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7748 <https://github.com/numba/numba/pull/7748>`_: Fix #7713: Ensure _prng_random_hash return has correct bitwidth (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7749 <https://github.com/numba/numba/pull/7749>`_: Refactor threading layer priority tests to not use stdout/stderr (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7752 <https://github.com/numba/numba/pull/7752>`_: Fix #7751: Use original filename for array exprs (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7755 <https://github.com/numba/numba/pull/7755>`_: CUDA: Deprecate support for CC < 5.3 and CTK < 10.2 (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7763 <https://github.com/numba/numba/pull/7763>`_: Update Read the Docs configuration (automatic) (`readthedocs-assistant <https://github.com/readthedocs-assistant>`_)
* PR `#7764 <https://github.com/numba/numba/pull/7764>`_: Add dbg_optnone and dbg_extend_lifetimes flags (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7771 <https://github.com/numba/numba/pull/7771>`_: Move function unique ID to abi-tags (`stuartarchibald <https://github.com/stuartarchibald>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7772 <https://github.com/numba/numba/pull/7772>`_: CUDA: Add Support to Creating `StructModel` Array (`Michael Wang <https://github.com/isVoid>`_)
* PR `#7776 <https://github.com/numba/numba/pull/7776>`_: Updates coverage.py config (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7777 <https://github.com/numba/numba/pull/7777>`_: Remove reference existing issue from GH template. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7778 <https://github.com/numba/numba/pull/7778>`_: Remove long deprecated flags from the CLI. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7780 <https://github.com/numba/numba/pull/7780>`_: Fix sets with reference counted items (`Benjamin Graham <https://github.com/benwilliamgraham>`_)
* PR `#7782 <https://github.com/numba/numba/pull/7782>`_: adding reminder to check on deprecations (`esc <https://github.com/esc>`_)
* PR `#7783 <https://github.com/numba/numba/pull/7783>`_: remove upper limit on Python version (`esc <https://github.com/esc>`_)
* PR `#7786 <https://github.com/numba/numba/pull/7786>`_: Remove dependency on intel-openmp for OSX (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7788 <https://github.com/numba/numba/pull/7788>`_: Avoid issue with DI gen for arrayexprs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7796 <https://github.com/numba/numba/pull/7796>`_: update change-log for 0.55.1 (`esc <https://github.com/esc>`_)
* PR `#7797 <https://github.com/numba/numba/pull/7797>`_: prune README (`esc <https://github.com/esc>`_)
* PR `#7799 <https://github.com/numba/numba/pull/7799>`_: update the release checklist post 0.55.1 (`esc <https://github.com/esc>`_)
* PR `#7801 <https://github.com/numba/numba/pull/7801>`_: add sdist command and umask reminder (`esc <https://github.com/esc>`_)
* PR `#7804 <https://github.com/numba/numba/pull/7804>`_: update local references from master -> main (`esc <https://github.com/esc>`_)
* PR `#7805 <https://github.com/numba/numba/pull/7805>`_: Enhance source line finding logic for debuginfo (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7809 <https://github.com/numba/numba/pull/7809>`_: Updates the gdb configuration to accept a binary name or a path. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7813 <https://github.com/numba/numba/pull/7813>`_: Extend parfors test timeout for aarch64. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7814 <https://github.com/numba/numba/pull/7814>`_: CUDA Dispatcher refactor (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7815 <https://github.com/numba/numba/pull/7815>`_: CUDA Dispatcher refactor 2: inherit from `dispatcher.Dispatcher` (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7817 <https://github.com/numba/numba/pull/7817>`_: Update intersphinx URLs for NumPy and llvmlite. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7823 <https://github.com/numba/numba/pull/7823>`_: Add renamed vars to callee scope such that it is self consistent. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7829 <https://github.com/numba/numba/pull/7829>`_: CUDA: Support `Enum/IntEnum` in Kernel (`Michael Wang <https://github.com/isVoid>`_)
* PR `#7833 <https://github.com/numba/numba/pull/7833>`_: Add version support information table to docs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7835 <https://github.com/numba/numba/pull/7835>`_: Fix pickling error when module cannot be imported (`idorrington <https://github.com/idorrington>`_)
* PR `#7836 <https://github.com/numba/numba/pull/7836>`_: min() and max() support for np.datetime and np.timedelta (`Benjamin Graham <https://github.com/benwilliamgraham>`_)
* PR `#7837 <https://github.com/numba/numba/pull/7837>`_: Initial refactoring of parfor reduction lowering  (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7845 <https://github.com/numba/numba/pull/7845>`_: change time.time() to time.perf_counter() in docs (`Nopileos2 <https://github.com/Nopileos2>`_)
* PR `#7846 <https://github.com/numba/numba/pull/7846>`_: Fix CUDA enum vectorize test on Windows (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7848 <https://github.com/numba/numba/pull/7848>`_: Support for int * list (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7850 <https://github.com/numba/numba/pull/7850>`_: CUDA: Pass `fastmath` compiler flag down to `compile_ptx` and `compile_device`; Improve `fastmath` tests (`Michael Wang <https://github.com/isVoid>`_)
* PR `#7855 <https://github.com/numba/numba/pull/7855>`_: Ensure np.argmin/no.argmax return type is intp (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7858 <https://github.com/numba/numba/pull/7858>`_: CUDA: Deprecate `ptx` Attribute and Update Tests (`Graham Markall <https://github.com/gmarkall>`_ `Michael Wang <https://github.com/isVoid>`_)
* PR `#7861 <https://github.com/numba/numba/pull/7861>`_: Fix a spelling mistake in README (`Zizheng Guo <https://github.com/gzz2000>`_)
* PR `#7864 <https://github.com/numba/numba/pull/7864>`_: Fix cross_iter_dep check. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7865 <https://github.com/numba/numba/pull/7865>`_: Remove add_user_function (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7866 <https://github.com/numba/numba/pull/7866>`_: Support for large numbers of args/kws with Python 3.10 (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7878 <https://github.com/numba/numba/pull/7878>`_: CUDA: Remove some deprecated support, add CC 8.6 and 8.7 (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7893 <https://github.com/numba/numba/pull/7893>`_: Use uuid.uuid4() as the key in serialization. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7895 <https://github.com/numba/numba/pull/7895>`_: Remove use of `llvmlite.llvmpy` (`Andre Masella <https://github.com/apmasell>`_)
* PR `#7898 <https://github.com/numba/numba/pull/7898>`_: Skip test_ptds under cuda-memcheck (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7901 <https://github.com/numba/numba/pull/7901>`_: Pyston compatibility for the test suite (`Kevin Modzelewski <https://github.com/kmod>`_)
* PR `#7904 <https://github.com/numba/numba/pull/7904>`_: Support m1 (`esc <https://github.com/esc>`_)
* PR `#7911 <https://github.com/numba/numba/pull/7911>`_: added sys import (`Nightfurex <https://github.com/Nightfurex>`_)
* PR `#7915 <https://github.com/numba/numba/pull/7915>`_: CUDA: Fix test checking debug info rendering. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7918 <https://github.com/numba/numba/pull/7918>`_: Add JIT examples to CUDA docs (`brandon-b-miller <https://github.com/brandon-b-miller>`_ `Graham Markall <https://github.com/gmarkall>`_)
* PR `#7919 <https://github.com/numba/numba/pull/7919>`_: Disallow //= reductions in pranges. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7924 <https://github.com/numba/numba/pull/7924>`_: Retain non-modified index tuple components. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7939 <https://github.com/numba/numba/pull/7939>`_: Fix rendering in feature request template. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7940 <https://github.com/numba/numba/pull/7940>`_: Implemented `np.allclose` in `numba/np/arraymath.py` (`Gagandeep Singh <https://github.com/czgdp1807>`_)
* PR `#7941 <https://github.com/numba/numba/pull/7941>`_: Remove debug dump output from closure inlining pass. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7946 <https://github.com/numba/numba/pull/7946>`_: instructions for creating a build environment were outdated (`esc <https://github.com/esc>`_)
* PR `#7949 <https://github.com/numba/numba/pull/7949>`_: Add Cuda Vector Types (`Michael Wang <https://github.com/isVoid>`_)
* PR `#7950 <https://github.com/numba/numba/pull/7950>`_: mission statement (`esc <https://github.com/esc>`_)
* PR `#7956 <https://github.com/numba/numba/pull/7956>`_: Stop using pip for 3.10 on public ci (Revert "start testing Python 3.10 on public CI") (`esc <https://github.com/esc>`_)
* PR `#7957 <https://github.com/numba/numba/pull/7957>`_: Use cloudpickle for disk caches (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7958 <https://github.com/numba/numba/pull/7958>`_: `numpy.clip` accept `numpy.array` for `a_min`, `a_max` (`Gagandeep Singh <https://github.com/czgdp1807>`_)
* PR `#7959 <https://github.com/numba/numba/pull/7959>`_: Permit a new array model to have a super set of array model fields. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7961 <https://github.com/numba/numba/pull/7961>`_: `numba.typed.typeddict.Dict.get` uses `castedkey` to avoid returning default value even if the key is present (`Gagandeep Singh <https://github.com/czgdp1807>`_)
* PR `#7963 <https://github.com/numba/numba/pull/7963>`_: remove the roadmap from the sphinx based docs (`esc <https://github.com/esc>`_)
* PR `#7964 <https://github.com/numba/numba/pull/7964>`_: Support for large constant dictionaries in Python 3.10 (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7965 <https://github.com/numba/numba/pull/7965>`_: Use uuid4 instead of PID in cache temp name to prevent collisions. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7971 <https://github.com/numba/numba/pull/7971>`_: lru cache for configure call (`Tingkai Liu <https://github.com/TK-21st>`_)
* PR `#7972 <https://github.com/numba/numba/pull/7972>`_: Fix fp16 support for cuda shared array (`Michael Collison <https://github.com/testhound>`_ `Graham Markall <https://github.com/gmarkall>`_)
* PR `#7986 <https://github.com/numba/numba/pull/7986>`_: Small caching refactor to support target cache implementations (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7994 <https://github.com/numba/numba/pull/7994>`_: Supporting multidimensional arrays in quick sort (`Gagandeep Singh <https://github.com/czgdp1807>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7996 <https://github.com/numba/numba/pull/7996>`_: Fix binding logic in `@overload_glue`. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7999 <https://github.com/numba/numba/pull/7999>`_: Remove `@overload_glue` for NumPy allocators. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8003 <https://github.com/numba/numba/pull/8003>`_: Add np.broadcast_shapes (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#8004 <https://github.com/numba/numba/pull/8004>`_: CUDA fixes for Windows (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8014 <https://github.com/numba/numba/pull/8014>`_: Fix support for {real,imag} array attrs in Parfors. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8016 <https://github.com/numba/numba/pull/8016>`_: [Docs] [Very Minor] Make `numba.jit` boundscheck doc line consistent (`Kyle Martin <https://github.com/martinky24>`_)
* PR `#8017 <https://github.com/numba/numba/pull/8017>`_: Update FAQ to include details about using debug-only option (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#8027 <https://github.com/numba/numba/pull/8027>`_: Support for NumPy 1.22 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8031 <https://github.com/numba/numba/pull/8031>`_: Support for Numpy BitGenerators PR#1 - Core Generator Support (`Kaustubh <https://github.com/kc611>`_)
* PR `#8035 <https://github.com/numba/numba/pull/8035>`_: Fix a couple of typos RE implementation (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8037 <https://github.com/numba/numba/pull/8037>`_: CUDA self-recursion tests (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8044 <https://github.com/numba/numba/pull/8044>`_: Make Python 3.10 kwarg peephole less restrictive (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#8046 <https://github.com/numba/numba/pull/8046>`_: Fix caching test failures (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8049 <https://github.com/numba/numba/pull/8049>`_: support str(bool) syntax (`LI Da <https://github.com/dlee992>`_)
* PR `#8052 <https://github.com/numba/numba/pull/8052>`_: Ensure pthread is linked in when building for ppc64le. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8056 <https://github.com/numba/numba/pull/8056>`_: Move caching tests from test_dispatcher to test_caching (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8057 <https://github.com/numba/numba/pull/8057>`_: Fix coverage checking (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8064 <https://github.com/numba/numba/pull/8064>`_: Rename "nb:run_pass" to "numba:run_pass" and document it. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8065 <https://github.com/numba/numba/pull/8065>`_: Fix PyLowering mishandling starargs (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8068 <https://github.com/numba/numba/pull/8068>`_: update changelog for 0.55.2 (`esc <https://github.com/esc>`_)
* PR `#8077 <https://github.com/numba/numba/pull/8077>`_: change return type of np.broadcast_shapes to a tuple (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#8080 <https://github.com/numba/numba/pull/8080>`_: Fix windows test failure due to timeout when the machine is slow poss… (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8081 <https://github.com/numba/numba/pull/8081>`_: Fix erroneous array count in parallel gufunc kernel generation. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8089 <https://github.com/numba/numba/pull/8089>`_: Support on-disk caching in the CUDA target (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8097 <https://github.com/numba/numba/pull/8097>`_: Exclude libopenblas 0.3.20 on osx-arm64 (`esc <https://github.com/esc>`_)
* PR `#8099 <https://github.com/numba/numba/pull/8099>`_: Fix Py_DECREF use in case of error state (for devicearray). (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8102 <https://github.com/numba/numba/pull/8102>`_: Combine numpy run_constrained in meta.yaml to the run requirements (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8109 <https://github.com/numba/numba/pull/8109>`_: Pin TBB support with respect to incompatible 2021.6 API. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8118 <https://github.com/numba/numba/pull/8118>`_: Update release checklists post 0.55.2 (`esc <https://github.com/esc>`_)
* PR `#8123 <https://github.com/numba/numba/pull/8123>`_: Fix CUDA print tests on Windows (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8124 <https://github.com/numba/numba/pull/8124>`_: Add explicit checks to all allocators in the NRT. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8126 <https://github.com/numba/numba/pull/8126>`_: Mark gufuncs as having mutable outputs (`Andre Masella <https://github.com/apmasell>`_)
* PR `#8133 <https://github.com/numba/numba/pull/8133>`_: Fix #8132. Regression in Record.make_c_struct for handling nestedarray (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8137 <https://github.com/numba/numba/pull/8137>`_: CUDA: Fix #7806, Division by zero stops the kernel (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8142 <https://github.com/numba/numba/pull/8142>`_: CUDA: Fix some missed changes from dropping 9.2 (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8144 <https://github.com/numba/numba/pull/8144>`_: Fix NumPy capitalisation in docs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8145 <https://github.com/numba/numba/pull/8145>`_: Allow ufunc builder to use previously JITed function (`Andre Masella <https://github.com/apmasell>`_)
* PR `#8151 <https://github.com/numba/numba/pull/8151>`_: pin NumPy to build 0 of 1.19.2 on public CI (`esc <https://github.com/esc>`_)
* PR `#8163 <https://github.com/numba/numba/pull/8163>`_: CUDA: Remove context query in launch config (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8165 <https://github.com/numba/numba/pull/8165>`_: Restrict strace based tests to be linux only via support feature. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8170 <https://github.com/numba/numba/pull/8170>`_: CUDA: Fix missing space in low occupancy warning (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8175 <https://github.com/numba/numba/pull/8175>`_: make build and upload order consistent (`esc <https://github.com/esc>`_)
* PR `#8181 <https://github.com/numba/numba/pull/8181>`_: Fix various typos (`luzpaz <https://github.com/luzpaz>`_)
* PR `#8187 <https://github.com/numba/numba/pull/8187>`_: Update CHANGE_LOG for 0.55.2 (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_)
* PR `#8189 <https://github.com/numba/numba/pull/8189>`_: updated version support information for 0.55.2/0.57 (`esc <https://github.com/esc>`_)
* PR `#8191 <https://github.com/numba/numba/pull/8191>`_: CUDA: Update deprecation notes for 0.56. (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8192 <https://github.com/numba/numba/pull/8192>`_: Update CHANGE_LOG for 0.56.0 (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8195 <https://github.com/numba/numba/pull/8195>`_: Make the workqueue threading backend once again fork safe. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8196 <https://github.com/numba/numba/pull/8196>`_: Fix numerical tolerance in parfors caching test. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8197 <https://github.com/numba/numba/pull/8197>`_: Fix `isinstance` warning check test. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8203 <https://github.com/numba/numba/pull/8203>`_: pin llvmlite 0.39 for public CI builds (`esc <https://github.com/esc>`_)
* PR `#8255 <https://github.com/numba/numba/pull/8255>`_: CUDA: Make numba.cuda.tests.doc_examples.ffi a module to fix #8252 (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#8274 <https://github.com/numba/numba/pull/8274>`_: Update version support table doc for 0.56. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8275 <https://github.com/numba/numba/pull/8275>`_: Update CHANGE_LOG for 0.56.0 final (`stuartarchibald <https://github.com/stuartarchibald>`_)

Authors:

* `Andre Masella <https://github.com/apmasell>`_
* `Benjamin Graham <https://github.com/benwilliamgraham>`_
* `brandon-b-miller <https://github.com/brandon-b-miller>`_
* `Brandon T. Willard <https://github.com/brandonwillard>`_
* `Gagandeep Singh <https://github.com/czgdp1807>`_
* `Dhruv Patel <https://github.com/DhruvPatel01>`_
* `LI Da <https://github.com/dlee992>`_
* `Todd A. Anderson <https://github.com/DrTodd13>`_
* `Ethan Pronovost <https://github.com/EPronovost>`_
* `esc <https://github.com/esc>`_
* `Tobias Sargeant <https://github.com/folded>`_
* `Graham Markall <https://github.com/gmarkall>`_
* `Guilherme Leobas <https://github.com/guilhermeleobas>`_
* `Zizheng Guo <https://github.com/gzz2000>`_
* `Hadia Ahmed <https://github.com/hadia206>`_
* `idorrington <https://github.com/idorrington>`_
* `Michael Wang <https://github.com/isVoid>`_
* `Kaustubh <https://github.com/kc611>`_
* `Kevin Modzelewski <https://github.com/kmod>`_
* `luk-f-a <https://github.com/luk-f-a>`_
* `luzpaz <https://github.com/luzpaz>`_
* `Kyle Martin <https://github.com/martinky24>`_
* `Nightfurex <https://github.com/Nightfurex>`_
* `Nick Riasanovsky <https://github.com/njriasan>`_
* `Nopileos2 <https://github.com/Nopileos2>`_
* `Ray Bell <https://github.com/raybellwaves>`_
* `readthedocs-assistant <https://github.com/readthedocs-assistant>`_
* `Rishi Kulkarni <https://github.com/rishi-kulkarni>`_
* `Sterling Baird <https://github.com/sgbaird>`_
* `Siu Kwan Lam <https://github.com/sklam>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_
* `Stepan Rakitin <https://github.com/svrakitin>`_
* `Michael Collison <https://github.com/testhound>`_
* `Tingkai Liu <https://github.com/TK-21st>`_

Version 0.55.2 (25 May, 2022)
-----------------------------

This is a maintenance release to support NumPy 1.22 and Apple M1.

Pull-Requests:

* PR `#8067 <https://github.com/numba/numba/pull/8067>`_: Backport #8027: Support for NumPy 1.22 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8069 <https://github.com/numba/numba/pull/8069>`_: Install llvmlite 0.38 for Numba 0.55.* (`esc <https://github.com/esc>`_)
* PR `#8075 <https://github.com/numba/numba/pull/8075>`_: update max NumPy for 0.55.2 (`esc <https://github.com/esc>`_)
* PR `#8078 <https://github.com/numba/numba/pull/8078>`_: Backport #7804: update local references from master -> main (`esc <https://github.com/esc>`_)
* PR `#8082 <https://github.com/numba/numba/pull/8082>`_: Backport #8080: fix windows failure due to timeout (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8084 <https://github.com/numba/numba/pull/8084>`_: Pin meta.yaml to llvmlite 0.38 series (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8093 <https://github.com/numba/numba/pull/8093>`_: Backport #7904: Support m1 (`esc <https://github.com/esc>`_)
* PR `#8094 <https://github.com/numba/numba/pull/8094>`_: Backport #8052 Ensure pthread is linked in when building for ppc64le. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8098 <https://github.com/numba/numba/pull/8098>`_: Backport #8097: Exclude libopenblas 0.3.20 on osx-arm64 (`esc <https://github.com/esc>`_)
* PR `#8100 <https://github.com/numba/numba/pull/8100>`_: Backport #7786 for 0.55.2: Remove dependency on intel-openmp for OSX  (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#8103 <https://github.com/numba/numba/pull/8103>`_: Backport #8102 to fix numpy requirements (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#8114 <https://github.com/numba/numba/pull/8114>`_: Backport #8109 Pin TBB support with respect to incompatible 2021.6 API. (`stuartarchibald <https://github.com/stuartarchibald>`_)

Total PRs: 12

Authors:

* `esc <https://github.com/esc>`_
* `Siu Kwan Lam <https://github.com/sklam>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_

Total authors: 3

Version 0.55.1 (27 January, 2022)
---------------------------------

This is a bugfix release that closes all the remaining issues from the
accelerated release of 0.55.0 and also any release critical regressions
discovered since then.

CUDA target deprecation notices:

* Support for CUDA toolkits < 10.2 is deprecated and will be removed in Numba
  0.56.
* Support for devices with Compute Capability < 5.3 is deprecated and will be
  removed in Numba 0.56.


Pull-Requests:

* PR `#7755 <https://github.com/numba/numba/pull/7755>`_: CUDA: Deprecate support for CC < 5.3 and CTK < 10.2 (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7749 <https://github.com/numba/numba/pull/7749>`_: Refactor threading layer priority tests to not use stdout/stderr (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7744 <https://github.com/numba/numba/pull/7744>`_: Fix issues with locating/parsing source during DebugInfo emission. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7712 <https://github.com/numba/numba/pull/7712>`_: Fixing issue 7693 (`Graham Markall <https://github.com/gmarkall>`_ `luk-f-a <https://github.com/luk-f-a>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7729 <https://github.com/numba/numba/pull/7729>`_: Handle Omitted/OmittedArgDataModel in DI generation. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7788 <https://github.com/numba/numba/pull/7788>`_: Avoid issue with DI gen for arrayexprs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7752 <https://github.com/numba/numba/pull/7752>`_: Fix #7751: Use original filename for array exprs (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7748 <https://github.com/numba/numba/pull/7748>`_: Fix #7713: Ensure _prng_random_hash return has correct bitwidth (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7745 <https://github.com/numba/numba/pull/7745>`_: Fix the release year for Numba 0.55 change log entry. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7740 <https://github.com/numba/numba/pull/7740>`_: CUDA Python 11.6 support (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7724 <https://github.com/numba/numba/pull/7724>`_: Update URLs in error messages to refer to RTD docs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7709 <https://github.com/numba/numba/pull/7709>`_: CUDA: Fixes missing type annotation pass following #7704 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7704 <https://github.com/numba/numba/pull/7704>`_: Move the type annotation pass to post legalization. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7619 <https://github.com/numba/numba/pull/7619>`_: CUDA: Fix linking with PTX when compiling lazily (`Graham Markall <https://github.com/gmarkall>`_)

Authors:

* `Graham Markall <https://github.com/gmarkall>`_
* `luk-f-a <https://github.com/luk-f-a>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_

Version 0.55.0 (13 January, 2022)
---------------------------------

This release includes a significant number important dependency upgrades along
with a number of new features and bug fixes.

NOTE: Due to NumPy CVE-2021-33430 this release has bypassed the usual release
process so as to promptly provide a Numba release that supports NumPy 1.21. A
single release candidate (RC1) was made and a few issues were reported, these
are summarised as follows and will be fixed in a subsequent 0.55.1 release.

Known issues with this release:

* Incorrect result copying array-typed field of structured array (`#7693 <https://github.com/numba/numba/pull/7693>`_)
* Two issues in DebugInfo generation (`#7726 <https://github.com/numba/numba/pull/7726>`_, `#7730 <https://github.com/numba/numba/pull/7730>`_)
* Compilation failure for ``hash`` of floating point values on 32 bit Windows
  when using Python 3.10 (`#7713 <https://github.com/numba/numba/pull/7713>`_).

Highlights of core dependency upgrades:

* Support for Python 3.10
* Support for NumPy 1.21

Python language support enhancements:

* Experimental support for ``isinstance``.

NumPy features/enhancements:

The following functions are now supported:

* ``np.broadcast_to``
* ``np.float_power``
* ``np.cbrt``
* ``np.logspace``
* ``np.take_along_axis``
* ``np.average``
* ``np.argmin`` gains support for the ``axis`` kwarg.
* ``np.ndarray.astype`` gains support for types expressed as literal strings.

Highlights of core changes:

* For users of the Numba extension API, Numba now has a new error handling mode
  whereby it will treat all exceptions that do not inherit from
  ``numba.errors.NumbaException`` as a "hard error" and immediately unwind the
  stack. This makes it much easier to debug when writing ``@overload``\s etc
  from the extension API as there's now no confusion between Python errors and
  Numba errors. This feature can be enabled by setting the environment
  variable: ``NUMBA_CAPTURED_ERRORS='new_style'``.
* The threading layer selection priority can now be changed via the environment
  variable ``NUMBA_THREADING_LAYER_PRIORITY``.

Highlights of changes for the CUDA target:

* Support for NVIDIA's CUDA Python bindings.
* Support for 16-bit floating point numbers and their basic operations via
  intrinsics.
* Streams are provided in the ``Stream.async_done`` result, making it easier to
  implement asynchronous work queues.
* Support for structured types in device arrays, character sequences in NumPy
  arrays, and some array operations on nested arrays.
* Much underlying refactoring to align the CUDA target more closely with the
  CPU target, which lays the groudwork for supporting the high level extension
  API in CUDA in future releases.

Intel also kindly sponsored research and development into native debug (DWARF)
support and handling per-function compilation flags:

* Line number/location tracking is much improved.
* Numba's internal representation of containers (e.g. tuples, arrays) are now
  encoded as structures.
* Numba's per-function compilation flags are encoded into the ABI field of the
  mangled name of the function such that it's possible to compile and
  differentiate between versions of the same function with different flags set.

General deprecation notices:

* There are no new general deprecations.

CUDA target deprecation notices:

* There are no new CUDA target deprecations.

Version support/dependency changes:

* Python 3.10 is supported.
* NumPy version 1.21 is supported.
* The minimum supported NumPy version is raised to 1.18 for runtime (compilation
  however remains compatible with NumPy 1.11).


Pull-Requests:

* PR `#6075 <https://github.com/numba/numba/pull/6075>`_: add np.float_power and np.cbrt (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#7047 <https://github.com/numba/numba/pull/7047>`_: Support __hash__ for numpy.datetime64 (`Guilherme Leobas <https://github.com/guilhermeleobas>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7057 <https://github.com/numba/numba/pull/7057>`_: Fix #7041: Add charseq registry to CUDA target (`Graham Markall <https://github.com/gmarkall>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7082 <https://github.com/numba/numba/pull/7082>`_: Added Add/Sub between datetime64 array and timedelta64 scalar (`Nick Riasanovsky <https://github.com/njriasan>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7119 <https://github.com/numba/numba/pull/7119>`_: Add support for `np.broadcast_to` (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#7129 <https://github.com/numba/numba/pull/7129>`_: Add support for axis keyword argument to np.argmin() (`Itamar Turner-Trauring <https://github.com/itamarst>`_)
* PR `#7132 <https://github.com/numba/numba/pull/7132>`_: gh #7131 Support for astype with literal strings (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7177 <https://github.com/numba/numba/pull/7177>`_: Add debug infomation support based on datamodel. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7185 <https://github.com/numba/numba/pull/7185>`_: Add get_impl_key as abstract method to types.Callable (`Alexey Kozlov <https://github.com/kozlov-alexey>`_)
* PR `#7186 <https://github.com/numba/numba/pull/7186>`_: Add support for np.logspace. (`Guoqiang QI <https://github.com/guoqiangqi>`_)
* PR `#7189 <https://github.com/numba/numba/pull/7189>`_: CUDA: Skip IPC tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7190 <https://github.com/numba/numba/pull/7190>`_: CUDA: Fix test_pinned on Jetson (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7192 <https://github.com/numba/numba/pull/7192>`_: Fix missing import in array.argsort impl and add more tests. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7196 <https://github.com/numba/numba/pull/7196>`_: Fixes for lineinfo emission (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7197 <https://github.com/numba/numba/pull/7197>`_: don't post to python announce on the first RC (`esc <https://github.com/esc>`_)
* PR `#7202 <https://github.com/numba/numba/pull/7202>`_: Initial implementation of np.take_along_axis (`Itamar Turner-Trauring <https://github.com/itamarst>`_)
* PR `#7203 <https://github.com/numba/numba/pull/7203>`_: remove duplicate changelog entries (`esc <https://github.com/esc>`_)
* PR `#7216 <https://github.com/numba/numba/pull/7216>`_: Update CHANGE_LOG for 0.54.0rc2 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7219 <https://github.com/numba/numba/pull/7219>`_: bump llvmlite dependency to 0.38.0dev0 for Numba 0.55.0dev0 (`esc <https://github.com/esc>`_)
* PR `#7220 <https://github.com/numba/numba/pull/7220>`_: update release checklist post 0.54rc1+2 (`esc <https://github.com/esc>`_)
* PR `#7221 <https://github.com/numba/numba/pull/7221>`_: Show GPU UUIDs in cuda.detect() output (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7222 <https://github.com/numba/numba/pull/7222>`_: CUDA: Warn when debug=True and opt=True (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7223 <https://github.com/numba/numba/pull/7223>`_: Replace assertion errors on IR assumption violation (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7226 <https://github.com/numba/numba/pull/7226>`_: Add support for structured types in Device Arrays (`Michael Collison <https://github.com/testhound>`_)
* PR `#7227 <https://github.com/numba/numba/pull/7227>`_: FIX: Typo (`Srinath Kailasa <https://github.com/skailasa>`_)
* PR `#7230 <https://github.com/numba/numba/pull/7230>`_: PR #7171 bugfix only (`stuartarchibald <https://github.com/stuartarchibald>`_ `Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7234 <https://github.com/numba/numba/pull/7234>`_: add THREADING_LAYER_PRIORITY & NUMBA_THREADING_LAYER_PRIORITY (`Kolen Cheung <https://github.com/ickc>`_)
* PR `#7235 <https://github.com/numba/numba/pull/7235>`_: replace wordings of WIP by draft PR (`Kolen Cheung <https://github.com/ickc>`_)
* PR `#7236 <https://github.com/numba/numba/pull/7236>`_: CUDA: Skip managed alloc tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7237 <https://github.com/numba/numba/pull/7237>`_: fix a typo in a string (`Kolen Cheung <https://github.com/ickc>`_)
* PR `#7241 <https://github.com/numba/numba/pull/7241>`_: Set aliasing information for inplace_binops.. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7242 <https://github.com/numba/numba/pull/7242>`_: FIX: typo (`Srinath Kailasa <https://github.com/skailasa>`_)
* PR `#7244 <https://github.com/numba/numba/pull/7244>`_: Implement partial literal propagation pass (support 'isinstance') (`Guilherme Leobas <https://github.com/guilhermeleobas>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7247 <https://github.com/numba/numba/pull/7247>`_: Solve memory leak to fix issue #7210  (`Siu Kwan Lam <https://github.com/sklam>`_ `Graham Markall <https://github.com/gmarkall>`_ `ysheffer <https://github.com/ysheffer>`_)
* PR `#7251 <https://github.com/numba/numba/pull/7251>`_: Fix #6001: typed.List ignores ctor arguments with JIT disabled (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7256 <https://github.com/numba/numba/pull/7256>`_: Fix link to the discourse forum in README (`Kenichi Maehashi <https://github.com/kmaehashi>`_)
* PR `#7257 <https://github.com/numba/numba/pull/7257>`_: Use normal list constructor in List.__new__() (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7260 <https://github.com/numba/numba/pull/7260>`_: Support typed lists in `heapq` (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7263 <https://github.com/numba/numba/pull/7263>`_: Updated issue URL for error messages #7261 (`DeviousLab <https://github.com/DeviousLab>`_)
* PR `#7265 <https://github.com/numba/numba/pull/7265>`_: Fix linspace to use np.divide and clamp to stop. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7266 <https://github.com/numba/numba/pull/7266>`_: CUDA: Skip multi-GPU copy test with peer access disabled (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7267 <https://github.com/numba/numba/pull/7267>`_: Fix #7258. Bug in SROA optimization (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7271 <https://github.com/numba/numba/pull/7271>`_: Update 3rd party license text. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7272 <https://github.com/numba/numba/pull/7272>`_: Allow annotations in njit-ed functions (`LunarLanding <https://github.com/LunarLanding>`_)
* PR `#7273 <https://github.com/numba/numba/pull/7273>`_: Update CHANGE_LOG for 0.54.0rc3. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7283 <https://github.com/numba/numba/pull/7283>`_: Added NPM to Glossary and linked to mentions (`Nihal Shetty <https://github.com/nihalshetty-boop>`_)
* PR `#7285 <https://github.com/numba/numba/pull/7285>`_: CUDA: Fix OOB in test_kernel_arg (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7288 <https://github.com/numba/numba/pull/7288>`_: Handle cval as a np attr in stencil generation. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7294 <https://github.com/numba/numba/pull/7294>`_: Continuation of PR #7280, fixing lifetime of TBB task_scheduler_handle (`Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7296 <https://github.com/numba/numba/pull/7296>`_: Fix generator lowering not casting to the actual yielded type (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7298 <https://github.com/numba/numba/pull/7298>`_: Use CBC to pin GCC to 7 on most linux and 9 on aarch64. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7304 <https://github.com/numba/numba/pull/7304>`_: Continue PR#3655: add support for np.average (`Hadia Ahmed <https://github.com/hadia206>`_ `slnguyen <https://github.com/slnguyen>`_)
* PR `#7307 <https://github.com/numba/numba/pull/7307>`_: Prevent mutation of arrays in global tuples. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7309 <https://github.com/numba/numba/pull/7309>`_: Update MapConstraint to handle type coercion for typed.Dict correctly. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7312 <https://github.com/numba/numba/pull/7312>`_: Fix #7302. Workaround missing pthread problem on ppc64le (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7315 <https://github.com/numba/numba/pull/7315>`_: Link ELF obj as DSO for radare2 disassembly CFG (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7316 <https://github.com/numba/numba/pull/7316>`_: Use float64 for consistent typing in heapq tests. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7317 <https://github.com/numba/numba/pull/7317>`_: In TBB tsh test switch os.fork for mp fork ctx (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7319 <https://github.com/numba/numba/pull/7319>`_: Update CHANGE_LOG for 0.54.0 final. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7329 <https://github.com/numba/numba/pull/7329>`_: Improve documentation in reference to CUDA local memory (`Sterling Baird <https://github.com/sgbaird>`_)
* PR `#7330 <https://github.com/numba/numba/pull/7330>`_: Cuda matmul docs (`Sterling Baird <https://github.com/sgbaird>`_)
* PR `#7340 <https://github.com/numba/numba/pull/7340>`_: Add size_t and ssize_t types (`Bruce Merry <https://github.com/bmerry>`_)
* PR `#7345 <https://github.com/numba/numba/pull/7345>`_: Add check for ipykernel file in IPython cache locator (`Sahil Gupta <https://github.com/sahil1105>`_)
* PR `#7347 <https://github.com/numba/numba/pull/7347>`_: fix:updated url for error report and feature rquest using issue template (`DEBARGHA SAHA <https://github.com/Stark-developer01>`_)
* PR `#7349 <https://github.com/numba/numba/pull/7349>`_: Allow arbitrary walk-back in reduction nodes to find inplace_binop. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7359 <https://github.com/numba/numba/pull/7359>`_: Extend support for nested arrays inside numpy records (`Graham Markall <https://github.com/gmarkall>`_ `luk-f-a <https://github.com/luk-f-a>`_)
* PR `#7375 <https://github.com/numba/numba/pull/7375>`_: CUDA: Run doctests as part of numba.cuda.tests and fix test_cg (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7395 <https://github.com/numba/numba/pull/7395>`_: Fix #7394 and #6550 & Added test & improved error message (`MegaIng <https://github.com/MegaIng>`_)
* PR `#7397 <https://github.com/numba/numba/pull/7397>`_: Add option to catch only Numba `numba.core.errors` derived exceptions. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7398 <https://github.com/numba/numba/pull/7398>`_: Add support for arrayanalysis of tuple args. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7403 <https://github.com/numba/numba/pull/7403>`_: Fix for issue 7402: implement missing numpy ufunc interface (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#7404 <https://github.com/numba/numba/pull/7404>`_: fix typo in literal_unroll docs (`esc <https://github.com/esc>`_)
* PR `#7419 <https://github.com/numba/numba/pull/7419>`_: insert missing backtick in comment (`esc <https://github.com/esc>`_)
* PR `#7422 <https://github.com/numba/numba/pull/7422>`_: Update Omitted Type to use Hashable Values as Keys for Caching (`Nick Riasanovsky <https://github.com/njriasan>`_)
* PR `#7429 <https://github.com/numba/numba/pull/7429>`_: Update CHANGE_LOG for 0.54.1 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7432 <https://github.com/numba/numba/pull/7432>`_: add github release task to checklist (`esc <https://github.com/esc>`_)
* PR `#7440 <https://github.com/numba/numba/pull/7440>`_: Refactor TargetConfig naming. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7441 <https://github.com/numba/numba/pull/7441>`_: Permit any string as a key in literalstrkeydict type. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7442 <https://github.com/numba/numba/pull/7442>`_: Add some diagnostics to SVML test failures. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7443 <https://github.com/numba/numba/pull/7443>`_: Refactor template selection logic for targets. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7444 <https://github.com/numba/numba/pull/7444>`_: use correct variable name in closure (`esc <https://github.com/esc>`_)
* PR `#7447 <https://github.com/numba/numba/pull/7447>`_: cleanup Numba metadata (`esc <https://github.com/esc>`_)
* PR `#7453 <https://github.com/numba/numba/pull/7453>`_: CUDA: Provide stream in async_done result (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7456 <https://github.com/numba/numba/pull/7456>`_: Fix invalid codegen for #7451. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7457 <https://github.com/numba/numba/pull/7457>`_: Factor out target registry selection logic (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7459 <https://github.com/numba/numba/pull/7459>`_: Include compiler flags in symbol mangling (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7460 <https://github.com/numba/numba/pull/7460>`_: Add FP16 support for CUDA (`Michael Collison <https://github.com/testhound>`_ `Graham Markall <https://github.com/gmarkall>`_)
* PR `#7461 <https://github.com/numba/numba/pull/7461>`_: Support NVIDIA's CUDA Python bindings (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7465 <https://github.com/numba/numba/pull/7465>`_: Update changelog for 0.54.1 release (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7477 <https://github.com/numba/numba/pull/7477>`_: Fix unicode operator.eq handling of Optional types. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7479 <https://github.com/numba/numba/pull/7479>`_: CUDA: Print format string and warn for > 32 print() args (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7483 <https://github.com/numba/numba/pull/7483>`_: NumPy 1.21 support (`Sebastian Berg <https://github.com/seberg>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7484 <https://github.com/numba/numba/pull/7484>`_: Fixed outgoing link to nvidia documentation. (`Dhruv Patel <https://github.com/DhruvPatel01>`_)
* PR `#7493 <https://github.com/numba/numba/pull/7493>`_: Consolidate TLS stacks in target configuration (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7496 <https://github.com/numba/numba/pull/7496>`_: CUDA: Use a single dispatcher class for all kinds of functions (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7498 <https://github.com/numba/numba/pull/7498>`_: refactor  with-detection logic (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_)
* PR `#7499 <https://github.com/numba/numba/pull/7499>`_: Add build scripts for CUDA testing on gpuCI  (`Charles Blackmon-Luca <https://github.com/charlesbluca>`_ `Graham Markall <https://github.com/gmarkall>`_)
* PR `#7500 <https://github.com/numba/numba/pull/7500>`_: Update parallel.rst (`Julius Bier Kirkegaard <https://github.com/juliusbierk>`_)
* PR `#7506 <https://github.com/numba/numba/pull/7506>`_: Enhance Flags mangling/demangling (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7514 <https://github.com/numba/numba/pull/7514>`_: Fixup cuda debuginfo emission for 7177 (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7525 <https://github.com/numba/numba/pull/7525>`_: Make sure` demangle()` returns `str` type. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7538 <https://github.com/numba/numba/pull/7538>`_: Fix `@overload_glue` performance regression. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7539 <https://github.com/numba/numba/pull/7539>`_: Fix str decode issue from merge #7525/#7506 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7546 <https://github.com/numba/numba/pull/7546>`_: Fix handling of missing const key in LiteralStrKeyDict (`Siu Kwan Lam <https://github.com/sklam>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7547 <https://github.com/numba/numba/pull/7547>`_: Remove 32bit linux scipy installation. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7548 <https://github.com/numba/numba/pull/7548>`_: Correct evaluation order in assert statement (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7552 <https://github.com/numba/numba/pull/7552>`_: Prepend the inlined function name to inlined variables. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7557 <https://github.com/numba/numba/pull/7557>`_: Python3.10 v2 (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_)
* PR `#7560 <https://github.com/numba/numba/pull/7560>`_: Refactor with detection py310 (`Siu Kwan Lam <https://github.com/sklam>`_ `esc <https://github.com/esc>`_)
* PR `#7561 <https://github.com/numba/numba/pull/7561>`_: fix a typo (`Kolen Cheung <https://github.com/ickc>`_)
* PR `#7567 <https://github.com/numba/numba/pull/7567>`_: Update docs to note meetings are public. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7570 <https://github.com/numba/numba/pull/7570>`_: Update the docs and error message for errors when importing Numba. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7580 <https://github.com/numba/numba/pull/7580>`_: Fix #7507. catch `NotImplementedError` in `.get_function()`  (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7581 <https://github.com/numba/numba/pull/7581>`_: Add support for casting from int enums (`Michael Collison <https://github.com/testhound>`_)
* PR `#7583 <https://github.com/numba/numba/pull/7583>`_: Make numba.types.Optional __str__ less verbose. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7588 <https://github.com/numba/numba/pull/7588>`_: Fix casting of start/stop in linspace (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7591 <https://github.com/numba/numba/pull/7591>`_: Remove deprecations (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7596 <https://github.com/numba/numba/pull/7596>`_: Fix max symbol match length for r2 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7597 <https://github.com/numba/numba/pull/7597>`_: Update gdb docs for new DWARF enhancements. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7603 <https://github.com/numba/numba/pull/7603>`_: Fix list.insert() for refcounted values (`Ehsan Totoni <https://github.com/ehsantn>`_)
* PR `#7605 <https://github.com/numba/numba/pull/7605>`_: Fix TBB 2021 DSO names on OSX/Win and make TBB reporting consistent (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7606 <https://github.com/numba/numba/pull/7606>`_: Ensure a prescribed threading layer can load in CI. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7610 <https://github.com/numba/numba/pull/7610>`_: Fix #7609. Type should not be mutated. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7618 <https://github.com/numba/numba/pull/7618>`_: Fix the doc build: docutils 0.18 not compatible with pinned sphinx (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7626 <https://github.com/numba/numba/pull/7626>`_: Fix issues with package dependencies. (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_)
* PR `#7627 <https://github.com/numba/numba/pull/7627>`_: PR 7321 continued (`stuartarchibald <https://github.com/stuartarchibald>`_ `Eric Wieser <https://github.com/eric-wieser>`_)
* PR `#7628 <https://github.com/numba/numba/pull/7628>`_: Move to using windows-2019 images in Azure (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7632 <https://github.com/numba/numba/pull/7632>`_: Capture output in CUDA matmul doctest (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7636 <https://github.com/numba/numba/pull/7636>`_: Copy prange loop header to after the parfor. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7637 <https://github.com/numba/numba/pull/7637>`_: Increase the timeout on the SVML tests for loaded machines. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7645 <https://github.com/numba/numba/pull/7645>`_: In debuginfo, do not add noinline to functions marked alwaysinline (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7650 <https://github.com/numba/numba/pull/7650>`_: Move Azure builds to OSX 10.15 (`stuartarchibald <https://github.com/stuartarchibald>`_ `esc <https://github.com/esc>`_ `Siu Kwan Lam <https://github.com/sklam>`_)

Authors:

* `Bruce Merry <https://github.com/bmerry>`_
* `Charles Blackmon-Luca <https://github.com/charlesbluca>`_
* `DeviousLab <https://github.com/DeviousLab>`_
* `Dhruv Patel <https://github.com/DhruvPatel01>`_
* `Todd A. Anderson <https://github.com/DrTodd13>`_
* `Ehsan Totoni <https://github.com/ehsantn>`_
* `Eric Wieser <https://github.com/eric-wieser>`_
* `esc <https://github.com/esc>`_
* `Graham Markall <https://github.com/gmarkall>`_
* `Guilherme Leobas <https://github.com/guilhermeleobas>`_
* `Guoqiang QI <https://github.com/guoqiangqi>`_
* `Hadia Ahmed <https://github.com/hadia206>`_
* `Kolen Cheung <https://github.com/ickc>`_
* `Itamar Turner-Trauring <https://github.com/itamarst>`_
* `Julius Bier Kirkegaard <https://github.com/juliusbierk>`_
* `Kenichi Maehashi <https://github.com/kmaehashi>`_
* `Alexey Kozlov <https://github.com/kozlov-alexey>`_
* `luk-f-a <https://github.com/luk-f-a>`_
* `LunarLanding <https://github.com/LunarLanding>`_
* `MegaIng <https://github.com/MegaIng>`_
* `Nihal Shetty <https://github.com/nihalshetty-boop>`_
* `Nick Riasanovsky <https://github.com/njriasan>`_
* `Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_
* `Sahil Gupta <https://github.com/sahil1105>`_
* `Sebastian Berg <https://github.com/seberg>`_
* `Sterling Baird <https://github.com/sgbaird>`_
* `Srinath Kailasa <https://github.com/skailasa>`_
* `Siu Kwan Lam <https://github.com/sklam>`_
* `slnguyen <https://github.com/slnguyen>`_
* `DEBARGHA SAHA <https://github.com/Stark-developer01>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_
* `Michael Collison <https://github.com/testhound>`_
* `ysheffer <https://github.com/ysheffer>`_

Version 0.54.1 (7 October, 2021)
--------------------------------

This is a bugfix release for 0.54.0. It fixes a regression in structured array
type handling, a potential leak on initialization failure in the CUDA target, a
regression caused by Numba's vendored cloudpickle module resetting dynamic
classes and a few minor testing/infrastructure related problems.

* PR `#7348 <https://github.com/numba/numba/pull/7348>`_: test_inspect_cli: Decode exception with default (utf-8) codec (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7360 <https://github.com/numba/numba/pull/7360>`_: CUDA: Fix potential leaks when initialization fails (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7386 <https://github.com/numba/numba/pull/7386>`_: Ensure the NRT is initialized prior to use in external NRT tests. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7388 <https://github.com/numba/numba/pull/7388>`_: Patch cloudpickle to not reset dynamic class each time it is unpickled (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7393 <https://github.com/numba/numba/pull/7393>`_: skip azure pipeline test if file not present (`esc <https://github.com/esc>`_)
* PR `#7428 <https://github.com/numba/numba/pull/7428>`_: Fix regression #7355: cannot set items in structured array data types (`Siu Kwan Lam <https://github.com/sklam>`_)

Authors:

* `esc <https://github.com/esc>`_
* `Graham Markall <https://github.com/gmarkall>`_
* `Siu Kwan Lam <https://github.com/sklam>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_


Version 0.54.0 (19 August, 2021)
--------------------------------

This release includes a significant number of new features, important
refactoring, critical bug fixes and a number of dependency upgrades.

Python language support enhancements:

* Basic support for ``f-strings``.
* ``dict`` comprehensions are now supported.
* The ``sum`` built-in function is implemented.

NumPy features/enhancements:

The following functions are now supported:

  * ``np.clip``
  * ``np.iscomplex``
  * ``np.iscomplexobj``
  * ``np.isneginf``
  * ``np.isposinf``
  * ``np.isreal``
  * ``np.isrealobj``
  * ``np.isscalar``
  * ``np.random.dirichlet``
  * ``np.rot90``
  * ``np.swapaxes``

Also ``np.argmax`` has gained support for the ``axis`` keyword argument and it's
now possible to use ``0d`` NumPy arrays as scalars in ``__setitem__`` calls.

Internal changes:

* Debugging support through DWARF has been fixed and enhanced.
* Numba now optimises the way in which locals are emitted to help reduce time
  spent in LLVM's SROA passes.

CUDA target changes:

* Support for emitting ``lineinfo`` to be consumed by profiling tools such as
  Nsight Compute
* Improved fastmath code generation for various trig, division, and other
  functions
* Faster compilation using lazy addition of libdevice to compiled units
* Support for IPC on Windows
* Support for passing tuples to CUDA ufuncs
* Performance warnings:

  * When making implicit copies by calling a kernel on arrays in host memory
  * When occupancy is poor due to kernel or ufunc/gufunc configuration

* Support for implementing warp-aggregated intrinsics:

  * Using support for more CUDA functions: ``activemask()``, ``lanemask_lt()``
  * The ``ffs()`` function now works correctly!

* Support for ``@overload`` in the CUDA target

Intel kindly sponsored research and development that lead to a number of new
features and internal support changes:

* Dispatchers can now be retargetted to a new target via a user defined context
  manager.
* Support for custom NumPy array subclasses has been added (including an
  overloadable memory allocator).
* An inheritance based model for targets that permits targets to share
  ``@overload`` implementations.
* Per function compiler flags with inheritance behaviours.
* The extension API now has support for overloading class methods via the
  ``@overload_classmethod`` decorator.

Deprecations:

* The ``ROCm`` target (for AMD ROC GPUs) has been moved to an "unmaintained"
  status and a seperate repository stub has been created for it at:
  https://github.com/numba/numba-rocm

CUDA target deprecations and breaking changes:

* Relaxed strides checking is now the default when computing the contiguity of
  device arrays.
* The ``inspect_ptx()`` method is deprecated. For use cases that obtain PTX for
  further compilation outside of Numba, use ``compile_ptx()`` instead.
* Eager compilation of device functions (the case when ``device=True`` and a
  signature is provided) is deprecated.

Version support/dependency changes:

* LLVM 11 is now supported on all platforms via llvmlite.
* The minimum supported Python version is raised to 3.7.
* NumPy version 1.20 is supported.
* The minimum supported NumPy version is raised to 1.17 for runtime (compilation
  however remains compatible with NumPy 1.11).
* Vendor `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ `v1.6.0` --
  now used for all ``pickle`` operations.
* TBB >= 2021 is now supported and all prior versions are unsupported (not
  easily possible to maintain the ABI breaking changes).

Pull-Requests:

* PR `#4516 <https://github.com/numba/numba/pull/4516>`_: Make setitem accept 0d np-arrays (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#4610 <https://github.com/numba/numba/pull/4610>`_: Implement np.is* functions (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#5984 <https://github.com/numba/numba/pull/5984>`_: Handle idx and size unification in wrap_index manually. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#6468 <https://github.com/numba/numba/pull/6468>`_: Access ``replace_functions_map`` via PreParforPass instance (`Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_ `Reazul Hoque <https://github.com/reazulhoque>`_)
* PR `#6469 <https://github.com/numba/numba/pull/6469>`_: Add address space in pointer type (`Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_ `Reazul Hoque <https://github.com/reazulhoque>`_)
* PR `#6608 <https://github.com/numba/numba/pull/6608>`_: Support f-strings for common cases (`Ehsan Totoni <https://github.com/ehsantn>`_)
* PR `#6619 <https://github.com/numba/numba/pull/6619>`_: Improved fastmath code generation for trig, log, and exp/pow. (`Graham Markall <https://github.com/gmarkall>`_ `Michael Collison <https://github.com/testhound>`_)
* PR `#6681 <https://github.com/numba/numba/pull/6681>`_: Explicitly catch ``with..as`` and raise error. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6689 <https://github.com/numba/numba/pull/6689>`_: Fix setup.py build command detection (`Hannes Pahl <https://github.com/HPLegion>`_)
* PR `#6695 <https://github.com/numba/numba/pull/6695>`_: Enable negative indexing for cuda atomic operations (`Ashutosh Varma <https://github.com/ashutoshvarma>`_)
* PR `#6696 <https://github.com/numba/numba/pull/6696>`_: flake8: made more files flake8 compliant (`Ashutosh Varma <https://github.com/ashutoshvarma>`_)
* PR `#6698 <https://github.com/numba/numba/pull/6698>`_: Fix #6697: Wrong dtype when using np.asarray on DeviceNDArray (`Ashutosh Varma <https://github.com/ashutoshvarma>`_)
* PR `#6700 <https://github.com/numba/numba/pull/6700>`_: Add UUID to CUDA devices (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6709 <https://github.com/numba/numba/pull/6709>`_: Block matplotlib in test examples (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6718 <https://github.com/numba/numba/pull/6718>`_: doc: fix typo in rewrites.rst (extra iterates) (`Alexander-Makaryev <https://github.com/Alexander-Makaryev>`_)
* PR `#6720 <https://github.com/numba/numba/pull/6720>`_: Faster compile (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6730 <https://github.com/numba/numba/pull/6730>`_: Fix Typeguard error (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6731 <https://github.com/numba/numba/pull/6731>`_: Add CUDA-specific pipeline (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6735 <https://github.com/numba/numba/pull/6735>`_: CUDA: Don't parse IR for modules with llvmlite (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6736 <https://github.com/numba/numba/pull/6736>`_: Support for dict comprehension (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6742 <https://github.com/numba/numba/pull/6742>`_: Do not add overload function definitions to index. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6750 <https://github.com/numba/numba/pull/6750>`_: Bump to llvmlite 0.37 series (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6751 <https://github.com/numba/numba/pull/6751>`_: Suppress typeguard warnings that affect testing. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6753 <https://github.com/numba/numba/pull/6753>`_: The check for internal types in RewriteArrayExprs (`Alexander-Makaryev <https://github.com/Alexander-Makaryev>`_)
* PR `#6755 <https://github.com/numba/numba/pull/6755>`_: install llvmlite from numba/label/dev (`esc <https://github.com/esc>`_)
* PR `#6758 <https://github.com/numba/numba/pull/6758>`_: patch to compile _devicearray.cpp with c++11 (`esc <https://github.com/esc>`_)
* PR `#6760 <https://github.com/numba/numba/pull/6760>`_: Fix scheduler bug where it rounds to 0 divisions for a chunk. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#6762 <https://github.com/numba/numba/pull/6762>`_: Glue wrappers to create @overload from split typing and lowering. (`stuartarchibald <https://github.com/stuartarchibald>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6766 <https://github.com/numba/numba/pull/6766>`_: Fix DeviceNDArray null shape issue (`Michael Collison <https://github.com/testhound>`_)
* PR `#6769 <https://github.com/numba/numba/pull/6769>`_: CUDA: Replace ``CachedPTX`` and ``CachedCUFunction`` with ``CUDACodeLibrary`` functionality (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6776 <https://github.com/numba/numba/pull/6776>`_: Fix issue with TBB interface causing warnings and parfors counting them (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6779 <https://github.com/numba/numba/pull/6779>`_: Fix wrap_index type unification. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#6786 <https://github.com/numba/numba/pull/6786>`_: Fix gufunc kwargs support (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6788 <https://github.com/numba/numba/pull/6788>`_: Add support for fastmath 32-bit floating point divide (`Michael Collison <https://github.com/testhound>`_)
* PR `#6789 <https://github.com/numba/numba/pull/6789>`_: Fix warnings struct ref typeguard (`stuartarchibald <https://github.com/stuartarchibald>`_ `Siu Kwan Lam <https://github.com/sklam>`_ `esc <https://github.com/esc>`_)
* PR `#6794 <https://github.com/numba/numba/pull/6794>`_: refactor and move create_temp_module into numba.tests.support (`Alexander-Makaryev <https://github.com/Alexander-Makaryev>`_)
* PR `#6795 <https://github.com/numba/numba/pull/6795>`_: CUDA: Lazily add libdevice to compilation units  (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6798 <https://github.com/numba/numba/pull/6798>`_: CUDA: Add optional Driver API argument logging (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6799 <https://github.com/numba/numba/pull/6799>`_: Print Numba and llvmlite versions in sysinfo (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6800 <https://github.com/numba/numba/pull/6800>`_: Make a common standard API for querying ufunc impl (`Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6801 <https://github.com/numba/numba/pull/6801>`_: ParallelAccelerator no long will convert StaticSetItem to SetItem because record arrays require StaticSetItems. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#6802 <https://github.com/numba/numba/pull/6802>`_: Add lineinfo flag to PTX and SASS compilation (`Graham Markall <https://github.com/gmarkall>`_ `Max Katz <https://github.com/maxpkatz>`_)
* PR `#6804 <https://github.com/numba/numba/pull/6804>`_: added runtime version to ``numba -s`` (`Kalyan <https://github.com/rawwar>`_)
* PR `#6808 <https://github.com/numba/numba/pull/6808>`_: #3468 continued: Add support for ``np.clip`` (`Graham Markall <https://github.com/gmarkall>`_ `Aaron Russell Voelker <https://github.com/arvoelke>`_)
* PR `#6809 <https://github.com/numba/numba/pull/6809>`_: #3203 additional info in cuda detect (`Kalyan <https://github.com/rawwar>`_)
* PR `#6810 <https://github.com/numba/numba/pull/6810>`_: Fix tiny formatting error in ROC kernel docs (`Felix Divo <https://github.com/felixdivo>`_)
* PR `#6811 <https://github.com/numba/numba/pull/6811>`_: CUDA: Remove test of runtime being a supported version (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6813 <https://github.com/numba/numba/pull/6813>`_: Mostly CUDA: Replace llvmpy API usage with llvmlite APIs (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6814 <https://github.com/numba/numba/pull/6814>`_: Improving context stack (`stuartarchibald <https://github.com/stuartarchibald>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6818 <https://github.com/numba/numba/pull/6818>`_: CUDA: Support IPC on Windows (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6822 <https://github.com/numba/numba/pull/6822>`_: Add support for np.rot90 (`stuartarchibald <https://github.com/stuartarchibald>`_ `Daniel Nagel <https://github.com/braniii>`_)
* PR `#6829 <https://github.com/numba/numba/pull/6829>`_: Fix accuracy of np.arange and np.linspace (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6830 <https://github.com/numba/numba/pull/6830>`_: CUDA: Use relaxed strides checking to compute contiguity (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6833 <https://github.com/numba/numba/pull/6833>`_: Raise TypeError exception if numpy array is cast to scalar (`Michael Collison <https://github.com/testhound>`_)
* PR `#6834 <https://github.com/numba/numba/pull/6834>`_: Remove illegal "debug" kw argument (`Shaun Cutts <https://github.com/shaunc>`_)
* PR `#6836 <https://github.com/numba/numba/pull/6836>`_: CUDA: Documentation updates (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6840 <https://github.com/numba/numba/pull/6840>`_: CUDA: Remove items deprecated in 0.53 + simulator test fixes (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6841 <https://github.com/numba/numba/pull/6841>`_: CUDA: Fix source location on kernel entry and enable breakpoints to be set on kernels by mangled name (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6843 <https://github.com/numba/numba/pull/6843>`_: cross-referenced Array type in docs (`Kalyan <https://github.com/rawwar>`_)
* PR `#6844 <https://github.com/numba/numba/pull/6844>`_: CUDA: Remove NUMBAPRO env var warnings, envvars.py + other small tidy-ups (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6848 <https://github.com/numba/numba/pull/6848>`_: Ignore .ycm_extra_conf.py (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6849 <https://github.com/numba/numba/pull/6849>`_: Add __hash__ for IntEnum (`Hannes Pahl <https://github.com/HPLegion>`_)
* PR `#6850 <https://github.com/numba/numba/pull/6850>`_: Fix up more internal warnings (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6854 <https://github.com/numba/numba/pull/6854>`_: PR 6096 continued (`stuartarchibald <https://github.com/stuartarchibald>`_ `Ivan Butygin <https://github.com/Hardcode84>`_)
* PR `#6861 <https://github.com/numba/numba/pull/6861>`_: updated reference to hsa with roc (`Kalyan <https://github.com/rawwar>`_)
* PR `#6867 <https://github.com/numba/numba/pull/6867>`_: Update changelog for 0.53.1 (`esc <https://github.com/esc>`_)
* PR `#6869 <https://github.com/numba/numba/pull/6869>`_: Implement builtin sum() (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6870 <https://github.com/numba/numba/pull/6870>`_: Add support for dispatcher retargeting using with-context (`stuartarchibald <https://github.com/stuartarchibald>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6871 <https://github.com/numba/numba/pull/6871>`_: Force text-align:left when using Annotate (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#6873 <https://github.com/numba/numba/pull/6873>`_: docs: Update reference to @jitclass location (`David Nadlinger <https://github.com/dnadlinger>`_)
* PR `#6876 <https://github.com/numba/numba/pull/6876>`_: Add trailing slashes to dir paths in CODEOWNERS (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6877 <https://github.com/numba/numba/pull/6877>`_: Add doc for recent target extension features (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6878 <https://github.com/numba/numba/pull/6878>`_: CUDA: Support passing tuples to ufuncs (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6879 <https://github.com/numba/numba/pull/6879>`_: CUDA: NumPy and string dtypes for local and shared arrays (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6880 <https://github.com/numba/numba/pull/6880>`_: Add attribute lower_extension to CPUContext (`Reazul Hoque <https://github.com/reazulhoque>`_)
* PR `#6883 <https://github.com/numba/numba/pull/6883>`_: Add support of np.swapaxes #4074 (`Daniel Nagel <https://github.com/braniii>`_)
* PR `#6885 <https://github.com/numba/numba/pull/6885>`_: CUDA: Explicitly specify objmode + looplifting for jit functions in cuda.random (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6886 <https://github.com/numba/numba/pull/6886>`_: CUDA: Fix parallel testing for all testsuite submodules (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6888 <https://github.com/numba/numba/pull/6888>`_: Get overload to consider compiler flags in cache lookup (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6889 <https://github.com/numba/numba/pull/6889>`_: Address guvectorize too slow for cuda target (`Michael Collison <https://github.com/testhound>`_)
* PR `#6890 <https://github.com/numba/numba/pull/6890>`_: fixes #6884  (`Kalyan <https://github.com/rawwar>`_)
* PR `#6898 <https://github.com/numba/numba/pull/6898>`_: Work on overloading by hardware target. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6911 <https://github.com/numba/numba/pull/6911>`_: CUDA: Add support for activemask(), lanemask_lt(), and nanosleep() (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6912 <https://github.com/numba/numba/pull/6912>`_: Prevent use of varargs in closure calls. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6913 <https://github.com/numba/numba/pull/6913>`_: Add runtests option to gitdiff on the common ancestor (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6915 <https://github.com/numba/numba/pull/6915>`_: Update _Intrinsic for sphinx to capture the inner docstring (`Guilherme Leobas <https://github.com/guilhermeleobas>`_)
* PR `#6917 <https://github.com/numba/numba/pull/6917>`_: Add type conversion for StringLiteral to unicode_type and test. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6918 <https://github.com/numba/numba/pull/6918>`_: Start section on commonly encounted unsupported parfors code. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6924 <https://github.com/numba/numba/pull/6924>`_: CUDA: Fix ``ffs`` (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6928 <https://github.com/numba/numba/pull/6928>`_: Add support for axis keyword arg to numpy.argmax() (`stuartarchibald <https://github.com/stuartarchibald>`_ `Itamar Turner-Trauring <https://github.com/itamarst>`_)
* PR `#6929 <https://github.com/numba/numba/pull/6929>`_: Fix CI failure when gitpython is missing. (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6935 <https://github.com/numba/numba/pull/6935>`_: fixes broken link in numba-runtime.rst (`Kalyan <https://github.com/rawwar>`_)
* PR `#6936 <https://github.com/numba/numba/pull/6936>`_: CUDA: Implement support for PTDS globally (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6937 <https://github.com/numba/numba/pull/6937>`_: Fix memory leak in bytes boxing (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6940 <https://github.com/numba/numba/pull/6940>`_: Fix function resolution for intrinsics across hardware. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6941 <https://github.com/numba/numba/pull/6941>`_: ABC the target descriptor and make consistent throughout. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6944 <https://github.com/numba/numba/pull/6944>`_: CUDA: Support for ``@overload`` (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6945 <https://github.com/numba/numba/pull/6945>`_: Fix issue with array analysis tests needing scipy. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6948 <https://github.com/numba/numba/pull/6948>`_: Refactor registry init. (`stuartarchibald <https://github.com/stuartarchibald>`_ `Graham Markall <https://github.com/gmarkall>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6953 <https://github.com/numba/numba/pull/6953>`_: CUDA: Fix and deprecate ``inspect_ptx()``, fix NVVM option setup for device functions (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6958 <https://github.com/numba/numba/pull/6958>`_: Inconsistent behavior of reshape between numpy and numba/cuda device array (`Lauren Arnett <https://github.com/laurenarnett>`_)
* PR `#6961 <https://github.com/numba/numba/pull/6961>`_: Update overload glue to deal with typing_key (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6964 <https://github.com/numba/numba/pull/6964>`_: Move minimum supported Python version to 3.7 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6966 <https://github.com/numba/numba/pull/6966>`_: Fix issue with TBB test detecting forks from incorrect state. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6971 <https://github.com/numba/numba/pull/6971>`_: Fix CUDA ``@intrinsic`` use (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6977 <https://github.com/numba/numba/pull/6977>`_: Vendor cloudpickle (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#6978 <https://github.com/numba/numba/pull/6978>`_: Implement operator.contains for empty Tuples (`Brandon T. Willard <https://github.com/brandonwillard>`_)
* PR `#6981 <https://github.com/numba/numba/pull/6981>`_: Fix LLVM IR parsing error on use of ``np.bool_`` in globals (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6983 <https://github.com/numba/numba/pull/6983>`_: Support Optional types in ufuncs. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6985 <https://github.com/numba/numba/pull/6985>`_: Implement static set/get items on records with integer index (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6986 <https://github.com/numba/numba/pull/6986>`_: document release checklist (`esc <https://github.com/esc>`_)
* PR `#6989 <https://github.com/numba/numba/pull/6989>`_: update threading docs for function loading (`esc <https://github.com/esc>`_)
* PR `#6990 <https://github.com/numba/numba/pull/6990>`_: Refactor hardware extension API to refer to "target" instead. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6991 <https://github.com/numba/numba/pull/6991>`_: Move ROCm target status to "unmaintained". (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#6995 <https://github.com/numba/numba/pull/6995>`_: Resolve issue where nan was being assigned to int type numpy array (`Michael Collison <https://github.com/testhound>`_)
* PR `#6996 <https://github.com/numba/numba/pull/6996>`_: Add constant lowering support for `SliceType`s (`Brandon T. Willard <https://github.com/brandonwillard>`_)
* PR `#6997 <https://github.com/numba/numba/pull/6997>`_: CUDA: Remove catch of NotImplementedError in target.py (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#6999 <https://github.com/numba/numba/pull/6999>`_: Fix errors introduced by the cloudpickle patch (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7003 <https://github.com/numba/numba/pull/7003>`_: More mainline fixes (`stuartarchibald <https://github.com/stuartarchibald>`_ `Graham Markall <https://github.com/gmarkall>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7004 <https://github.com/numba/numba/pull/7004>`_: Test extending the CUDA target (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7007 <https://github.com/numba/numba/pull/7007>`_: Made stencil compilation not fail for arrays of conflicting types. (`MegaIng <https://github.com/MegaIng>`_)
* PR `#7008 <https://github.com/numba/numba/pull/7008>`_: Added support for np.random.dirichlet with all size arguments (`Rishi Kulkarni <https://github.com/rishi-kulkarni>`_)
* PR `#7016 <https://github.com/numba/numba/pull/7016>`_: Docs: Add DALI to list of CAI-supporting libraries (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7018 <https://github.com/numba/numba/pull/7018>`_: Remove cu{blas,sparse,rand,fft} from library checks (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7019 <https://github.com/numba/numba/pull/7019>`_: Support NumPy 1.20 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7020 <https://github.com/numba/numba/pull/7020>`_: Fix #7017. Adds util class PickleCallableByPath  (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7024 <https://github.com/numba/numba/pull/7024>`_: fixed llvmir usage in create_module method (`stuartarchibald <https://github.com/stuartarchibald>`_ `Kalyan <https://github.com/rawwar>`_)
* PR `#7027 <https://github.com/numba/numba/pull/7027>`_: Fix nrt debug print (`MegaIng <https://github.com/MegaIng>`_)
* PR `#7031 <https://github.com/numba/numba/pull/7031>`_: Fix inliner to use a single scope for all blocks (`Alexey Kozlov <https://github.com/kozlov-alexey>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7040 <https://github.com/numba/numba/pull/7040>`_: Add Github action to mark issues as stale (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7044 <https://github.com/numba/numba/pull/7044>`_: Fixes for LLVM 11 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7049 <https://github.com/numba/numba/pull/7049>`_: Make NumPy random module use @overload_glue (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7050 <https://github.com/numba/numba/pull/7050>`_: Add overload_classmethod (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7052 <https://github.com/numba/numba/pull/7052>`_: Fix string support in CUDA target (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7056 <https://github.com/numba/numba/pull/7056>`_: Change prange conversion approach to reuse header block. (`Todd A. Anderson <https://github.com/DrTodd13>`_)
* PR `#7061 <https://github.com/numba/numba/pull/7061>`_: Add ndarray allocator classmethod (`stuartarchibald <https://github.com/stuartarchibald>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7064 <https://github.com/numba/numba/pull/7064>`_: Testhound/host array performance warning (`Michael Collison <https://github.com/testhound>`_)
* PR `#7066 <https://github.com/numba/numba/pull/7066>`_: Fix #7065: Add expected exception messages for NumPy 1.20 to tests (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7068 <https://github.com/numba/numba/pull/7068>`_: Enhancing docs about PRNG seeding (`Jérome Eertmans <https://github.com/jeertmans>`_)
* PR `#7070 <https://github.com/numba/numba/pull/7070>`_: Improve the issue templates and pull request template. (`Guoqiang QI <https://github.com/guoqiangqi>`_)
* PR `#7080 <https://github.com/numba/numba/pull/7080>`_: Fix ``__eq__`` for Flags and cpu_options classes (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7087 <https://github.com/numba/numba/pull/7087>`_: Add note to docs about zero-initialization of variables. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7088 <https://github.com/numba/numba/pull/7088>`_: Initialize NUMBA_DEFAULT_NUM_THREADS with a batch scheduler aware value (`Thomas VINCENT <https://github.com/t20100>`_)
* PR `#7100 <https://github.com/numba/numba/pull/7100>`_: Replace deprecated call to cuDeviceComputeCapability (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7113 <https://github.com/numba/numba/pull/7113>`_: Temporarily disable debug env export. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7114 <https://github.com/numba/numba/pull/7114>`_: CUDA: Deprecate eager compilation of device functions (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7116 <https://github.com/numba/numba/pull/7116>`_: Fix various issues with dwarf emission: (`stuartarchibald <https://github.com/stuartarchibald>`_ `vlad-perevezentsev <https://github.com/vlad-perevezentsev>`_)
* PR `#7118 <https://github.com/numba/numba/pull/7118>`_: Remove print to stdout (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7121 <https://github.com/numba/numba/pull/7121>`_: Continue work on numpy subclasses (`Todd A. Anderson <https://github.com/DrTodd13>`_ `Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7122 <https://github.com/numba/numba/pull/7122>`_: Rtd/sphinx compat (`esc <https://github.com/esc>`_)
* PR `#7134 <https://github.com/numba/numba/pull/7134>`_: Move minimum LLVM version to 11. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7137 <https://github.com/numba/numba/pull/7137>`_: skip pycc test on Python 3.7 + macOS because of distutils issue (`esc <https://github.com/esc>`_)
* PR `#7138 <https://github.com/numba/numba/pull/7138>`_: Update the Azure default linux image to Ubuntu 18.04 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7141 <https://github.com/numba/numba/pull/7141>`_: Require llvmlite 0.37 as minimum supported. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7143 <https://github.com/numba/numba/pull/7143>`_: Update version checks in __init__ for np 1.17 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7145 <https://github.com/numba/numba/pull/7145>`_: Fix mainline (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7146 <https://github.com/numba/numba/pull/7146>`_: Fix ``inline_closurecall`` may not be imported (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7147 <https://github.com/numba/numba/pull/7147>`_: Revert "Workaround gitpython 3.1.18 dependency issue" (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7149 <https://github.com/numba/numba/pull/7149>`_: Fix issue in bytecode analysis where target and next are same. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7152 <https://github.com/numba/numba/pull/7152>`_: Fix iterators in CUDA (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7156 <https://github.com/numba/numba/pull/7156>`_: Fix ``ir_utils._max_label`` being updated incorrectly (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7160 <https://github.com/numba/numba/pull/7160>`_: Split parfors tests (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7161 <https://github.com/numba/numba/pull/7161>`_: Update README for 0.54 (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7162 <https://github.com/numba/numba/pull/7162>`_: CUDA: Fix linkage of device functions when compiling for debug (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7163 <https://github.com/numba/numba/pull/7163>`_: Split legalization pass to consider IR and features separately. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7165 <https://github.com/numba/numba/pull/7165>`_: Fix use of np.clip where out is not provided. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7189 <https://github.com/numba/numba/pull/7189>`_: CUDA: Skip IPC tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7190 <https://github.com/numba/numba/pull/7190>`_: CUDA: Fix test_pinned on Jetson (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7192 <https://github.com/numba/numba/pull/7192>`_: Fix missing import in array.argsort impl and add more tests. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7196 <https://github.com/numba/numba/pull/7196>`_: Fixes for lineinfo emission. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7203 <https://github.com/numba/numba/pull/7203>`_: remove duplicate changelog entries (`esc <https://github.com/esc>`_)
* PR `#7209 <https://github.com/numba/numba/pull/7209>`_: Clamp numpy (`esc <https://github.com/esc>`_)
* PR `#7216 <https://github.com/numba/numba/pull/7216>`_: Update CHANGE_LOG for 0.54.0rc2. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7223 <https://github.com/numba/numba/pull/7223>`_: Replace assertion errors on IR assumption violation (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7230 <https://github.com/numba/numba/pull/7230>`_: PR #7171 bugfix only (`Todd A. Anderson <https://github.com/DrTodd13>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7236 <https://github.com/numba/numba/pull/7236>`_: CUDA: Skip managed alloc tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7267 <https://github.com/numba/numba/pull/7267>`_: Fix #7258. Bug in SROA optimization (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7271 <https://github.com/numba/numba/pull/7271>`_: Update 3rd party license text. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7272 <https://github.com/numba/numba/pull/7272>`_: Allow annotations in njit-ed functions (`LunarLanding <https://github.com/LunarLanding>`_)
* PR `#7273 <https://github.com/numba/numba/pull/7273>`_: Update CHANGE_LOG for 0.54.0rc3. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7285 <https://github.com/numba/numba/pull/7285>`_: CUDA: Fix OOB in test_kernel_arg (`Graham Markall <https://github.com/gmarkall>`_)
* PR `#7294 <https://github.com/numba/numba/pull/7294>`_: Continuation of PR #7280, fixing lifetime of TBB task_scheduler_handle (`Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7298 <https://github.com/numba/numba/pull/7298>`_: Use CBC to pin GCC to 7 on most linux and 9 on aarch64. (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7312 <https://github.com/numba/numba/pull/7312>`_: Fix #7302. Workaround missing pthread problem on ppc64le (`Siu Kwan Lam <https://github.com/sklam>`_)
* PR `#7317 <https://github.com/numba/numba/pull/7317>`_: In TBB tsh test switch os.fork for mp fork ctx (`stuartarchibald <https://github.com/stuartarchibald>`_)
* PR `#7319 <https://github.com/numba/numba/pull/7319>`_: Update CHANGE_LOG for 0.54.0 final. (`stuartarchibald <https://github.com/stuartarchibald>`_)

Authors:

* `Alexander-Makaryev <https://github.com/Alexander-Makaryev>`_
* `Todd A. Anderson <https://github.com/DrTodd13>`_
* `Hannes Pahl <https://github.com/HPLegion>`_
* `Ivan Butygin <https://github.com/Hardcode84>`_
* `MegaIng <https://github.com/MegaIng>`_
* `Sergey Pokhodenko <https://github.com/PokhodenkoSA>`_
* `Aaron Russell Voelker <https://github.com/arvoelke>`_
* `Ashutosh Varma <https://github.com/ashutoshvarma>`_
* `Ben Greiner <https://github.com/bnavigator>`_
* `Brandon T. Willard <https://github.com/brandonwillard>`_
* `Daniel Nagel <https://github.com/braniii>`_
* `David Nadlinger <https://github.com/dnadlinger>`_
* `Ehsan Totoni <https://github.com/ehsantn>`_
* `esc <https://github.com/esc>`_
* `Felix Divo <https://github.com/felixdivo>`_
* `Graham Markall <https://github.com/gmarkall>`_
* `Guilherme Leobas <https://github.com/guilhermeleobas>`_
* `Guoqiang QI <https://github.com/guoqiangqi>`_
* `Itamar Turner-Trauring <https://github.com/itamarst>`_
* `Jérome Eertmans <https://github.com/jeertmans>`_
* `Alexey Kozlov <https://github.com/kozlov-alexey>`_
* `Lauren Arnett <https://github.com/laurenarnett>`_
* `LunarLanding <https://github.com/LunarLanding>`_
* `Max Katz <https://github.com/maxpkatz>`_
* `Kalyan <https://github.com/rawwar>`_
* `Reazul Hoque <https://github.com/reazulhoque>`_
* `Rishi Kulkarni <https://github.com/rishi-kulkarni>`_
* `Shaun Cutts <https://github.com/shaunc>`_
* `Siu Kwan Lam <https://github.com/sklam>`_
* `stuartarchibald <https://github.com/stuartarchibald>`_
* `Thomas VINCENT <https://github.com/t20100>`_
* `Michael Collison <https://github.com/testhound>`_
* `vlad-perevezentsev <https://github.com/vlad-perevezentsev>`_


Version 0.53.1 (25 March, 2021)
-------------------------------

This is a bugfix release for 0.53.0. It contains the following four
pull-requests which fix two critical regressions and two build failures
reported by the openSuSe team:

* PR #6826 Fix regression on gufunc serialization
* PR #6828 Fix regression in CUDA: Set stream in mapped and managed array
  device_setup
* PR #6837 Ignore warnings from packaging module when testing import behaviour.
* PR #6851 set non-reported llvm timing values to 0.0

Authors:

* Ben Greiner
* Graham Markall
* Siu Kwan Lam
* Stuart Archibald

Version 0.53.0 (11 March, 2021)
-------------------------------

This release continues to add new features, bug fixes and stability improvements
to Numba.

Highlights of core changes:

* Support for Python 3.9 (Stuart Archibald).
* Function sub-typing (Lucio Fernandez-Arjona).
* Initial support for dynamic ``gufuncs`` (i.e. from ``@guvectorize``)
  (Guilherme Leobas).
* Parallel Accelerator (``@njit(parallel=True)`` now supports Fortran ordered
  arrays (Todd A. Anderson and Siu Kwan Lam).

Intel also kindly sponsored research and development that lead to two new
features:

  * Exposing LLVM compilation pass timings for diagnostic purposes (Siu Kwan
    Lam).
  * An event system for broadcasting compiler events (Siu Kwan Lam).

Highlights of changes for the CUDA target:

* CUDA 11.2 onwards (versions of the toolkit using NVVM IR 1.6 / LLVM IR 7.0.1)
  are now supported (Graham Markall).
* A fast cube root function is added (Michael Collison).
* Support for atomic ``xor``, increment, decrement, exchange, are added, and
  compare-and-swap is extended to support 64-bit integers (Michael Collison).
* Addition of ``cuda.is_supported_version()`` to check if the CUDA runtime
  version is supported (Graham Markall).
* The CUDA dispatcher now shares infrastructure with the CPU dispatcher,
  improving launch times for lazily-compiled kernels (Graham Markall).
* The CUDA Array Interface is updated to version 3, with support for streams
  added (Graham Markall).
* Tuples and ``namedtuples`` can now be passed to kernels (Graham Markall).
* Initial support for Cooperative Groups is added, with support for Grid Groups
  and Grid Sync (Graham Markall and Nick White).
* Support for ``math.log2`` and ``math.remainder`` is added (Guilherme Leobas).

General deprecation notices:

* There are no new general deprecations.

CUDA target deprecation notices:

* CUDA support on macOS is deprecated with this release (it still works, it is
  just unsupported).
* The ``argtypes``, ``restypes``, and ``bind`` keyword arguments to the
  ``cuda.jit`` decorator, deprecated since 0.51.0, are removed
* The ``Device.COMPUTE_CAPABILITY`` property, deprecated since 2014, has been
  removed (use ``compute_capability`` instead).
* The ``to_host`` method of device arrays is removed (use ``copy_to_host``
  instead).

General Enhancements:

* PR #4769: objmode complex type spelling (Siu Kwan Lam)
* PR #5579: Function subtyping (Lucio Fernandez-Arjona)
* PR #5659: Add support for parfors creating 'F'ortran layout Numpy arrays.
  (Todd A. Anderson)
* PR #5936: Improve array analysis for user-defined data types. (Todd A.
  Anderson)
* PR #5938: Initial support for dynamic gufuncs (Guilherme Leobas)
* PR #5958: Making typed.List a typing Generic (Lucio Fernandez-Arjona)
* PR #6334: Support attribute access from other modules (Farah Hariri)
* PR #6373: Allow Dispatchers to be cached (Eric Wieser)
* PR #6519: Avoid unnecessary ir.Del generation and removal (Ehsan Totoni)
* PR #6545: Refactoring ParforDiagnostics (Elena Totmenina)
* PR #6560: Add LLVM pass timer (Siu Kwan Lam)
* PR #6573: Improve ``__str__`` for typed.List when invoked from IPython shell
  (Amin Sadeghi)
* PR #6575: Avoid temp variable assignments (Ehsan Totoni)
* PR #6578: Add support for numpy ``intersect1d`` and basic test cases
  (``@caljrobe``)
* PR #6579: Python 3.9 support. (Stuart Archibald)
* PR #6580: Store partial typing errors in compiler state (Ehsan Totoni)
* PR #6626: A simple event system to broadcast compiler events (Siu Kwan Lam)
* PR #6635: Try to resolve dynamic getitems as static post unroll transform.
  (Stuart Archibald)
* PR #6636: Adds llvm_lock event (Siu Kwan Lam)
* PR #6664: Adds tests for PR 5659 (Siu Kwan Lam)
* PR #6680: Allow getattr to work in objmode output type spec (Siu Kwan Lam)

Fixes:

* PR #6176: Remove references to deprecated numpy globals (Eric Wieser)
* PR #6374: Use Python 3 style OSError handling (Eric Wieser)
* PR #6402: Fix ``typed.Dict`` and ``typed.List`` crashing on parametrized types
  (Andreas Sodeur)
* PR #6403: Add ``types.ListType.key`` (Andreas Sodeur)
* PR #6410: Fixes issue #6386 (Danny Weitekamp)
* PR #6425: Fix unicode join for issue #6405 (Teugea Ioan-Teodor)
* PR #6437: Don't pass reduction variables known in an outer parfor to inner
  parfors when analyzing reductions. (Todd A. Anderson)
* PR #6453: Keep original variable names in metadata to improve diagnostics
  (Ehsan Totoni)
* PR #6454: FIX: Fixes for literals (Eric Larson)
* PR #6463: Bump llvmlite to 0.36 series (Stuart Archibald)
* PR #6466: Remove the misspelling of finalize_dynamic_globals (Sergey
  Pokhodenko)
* PR #6489: Improve the error message for unsupported Buffer in Buffer
  situation. (Stuart Archibald)
* PR #6503: Add test to ensure Numba imports without warnings. (Stuart
  Archibald)
* PR #6508: Defer requirements to setup.py (Siu Kwan Lam)
* PR #6521: Skip annotated jitclass test if typeguard is running. (Stuart
  Archibald)
* PR #6524: Fix typed.List return value (Lucio Fernandez-Arjona)
* PR #6562: Correcting typo in numba sysinfo output (Nick Sutcliffe)
* PR #6574: Run parfor fusion if 2 or more parfors (Ehsan Totoni)
* PR #6582: Fix typed dict error with uninitialized padding bytes  (Siu Kwan
  Lam)
* PR #6584: Remove jitclass from ``__init__`` ``__all__``. (Stuart Archibald)
* PR #6586: Run closure inlining ahead of branch pruning in case of nonlocal
  (Stuart Archibald)
* PR #6591: Fix inlineasm test failure. (Siu Kwan Lam)
* PR #6622: Fix 6534, handle unpack of assign-like tuples. (Stuart Archibald)
* PR #6652: Simplify PR-6334 (Siu Kwan Lam)
* PR #6653: Fix get_numba_envvar (Siu Kwan Lam)
* PR #6654: Fix #6632 support alternative dtype string spellings (Stuart
  Archibald)
* PR #6685: Add Python 3.9 to classifiers. (Stuart Archibald)
* PR #6693: patch to compile _devicearray.cpp with c++11 (Valentin Haenel)
* PR #6716: Consider assignment lhs live if used in rhs (Fixes #6715) (Ehsan
  Totoni)
* PR #6727: Avoid errors in array analysis for global tuples with non-int
  (Ehsan Totoni)
* PR #6733: Fix segfault and errors in #6668 (Siu Kwan Lam)
* PR #6741: Enable SSA in IR inliner (Ehsan Totoni)
* PR #6763: use an alternative constraint for the conda packages (Valentin
  Haenel)
* PR #6786: Fix gufunc kwargs support (Siu Kwan Lam)

CUDA Enhancements/Fixes:

* PR #5162: Specify synchronization semantics of CUDA Array Interface (Graham
  Markall)
* PR #6245: CUDA Cooperative grid groups (Graham Markall and Nick White)
* PR #6333: Remove dead ``_Kernel.__call__`` (Graham Markall)
* PR #6343: CUDA: Add support for passing tuples and namedtuples to kernels
  (Graham Markall)
* PR #6349: Refactor Dispatcher to remove unnecessary indirection (Graham
  Markall)
* PR #6358: Add log2 and remainder implementations for cuda (Guilherme Leobas)
* PR #6376: Added a fixed seed in test_atomics.py for issue #6370 (Teugea
  Ioan-Teodor)
* PR #6377: CUDA: Fix various issues in test suite (Graham Markall)
* PR #6409: Implement cuda atomic xor (Michael Collison)
* PR #6422: CUDA: Remove deprecated items, expect CUDA 11.1 (Graham Markall)
* PR #6427: Remove duplicate repeated definition of gufunc (Amit Kumar)
* PR #6432: CUDA: Use ``_dispatcher.Dispatcher`` as base Dispatcher class
  (Graham Markall)
* PR #6447: CUDA: Add get_regs_per_thread method to Dispatcher (Graham Markall)
* PR #6499: CUDA atomic increment, decrement, exchange and compare and swap
  (Michael Collison)
* PR #6510: CUDA: Make device array assignment synchronous where necessary
  (Graham Markall)
* PR #6517: CUDA: Add NVVM test of all 8-bit characters (Graham Markall)
* PR #6567: Refactor llvm replacement code into separate function (Michael
  Collison)
* PR #6642: Testhound/cuda cuberoot (Michael Collison)
* PR #6661: CUDA: Support NVVM70 / CUDA 11.2 (Graham Markall)
* PR #6663: Fix error caused by missing "-static" libraries defined for some
  platforms (Siu Kwan Lam)
* PR #6666: CUDA: Add a function to query whether the runtime version is
  supported. (Graham Markall)
* PR #6725: CUDA: Fix compile to PTX with debug for CUDA 11.2 (Graham Markall)

Documentation Updates:

* PR #5740: Add FAQ entry on how to create a MWR. (Stuart Archibald)
* PR #6346: DOC: add where to get dev builds from to FAQ  (Eyal Trabelsi)
* PR #6418: docs: use https for homepage (``@imba-tjd``)
* PR #6430: CUDA docs: Add RNG example with 3D grid and strided loops (Graham
  Markall)
* PR #6436: docs: remove typo in Deprecation Notices (Thibault Ballier)
* PR #6440: Add note about performance of typed containers from the interpreter.
  (Stuart Archibald)
* PR #6457: Link to read the docs instead of numba homepage (Hannes Pahl)
* PR #6470: Adding PyCon Sweden 2020 talk on numba (Ankit Mahato)
* PR #6472: Document ``numba.extending.is_jitted`` (Stuart Archibald)
* PR #6495: Fix typo in literal list docs. (Stuart Archibald)
* PR #6501: Add doc entry on Numba's limited resources and how to help. (Stuart
  Archibald)
* PR #6502: Add CODEOWNERS file. (Stuart Archibald)
* PR #6531: Update canonical URL. (Stuart Archibald)
* PR #6544: Minor typo / grammar fixes to 5 minute guide (Ollin Boer Bohan)
* PR #6599: docs: fix simple typo, consevatively -> conservatively (Tim Gates)
* PR #6609: Recommend miniforge instead of c4aarch64 (Isuru Fernando)
* PR #6671: Update environment creation example to python 3.8 (Lucio
  Fernandez-Arjona)
* PR #6676: Update hardware and software versions in various docs. (Stuart
  Archibald)
* PR #6682: Update deprecation notices for 0.53 (Stuart Archibald)

CI/Infrastructure Updates:

* PR #6458: Enable typeguard in CI (Siu Kwan Lam)
* PR #6500: Update bug and feature request templates. (Stuart Archibald)
* PR #6516: Fix RTD build by using conda. (Stuart Archibald)
* PR #6587: Add zenodo badge (Siu Kwan Lam)

Authors:

* Amin Sadeghi
* Amit Kumar
* Andreas Sodeur
* Ankit Mahato
* Chris Barnes
* Danny Weitekamp
* Ehsan Totoni (core dev)
* Eric Larson
* Eric Wieser
* Eyal Trabelsi
* Farah Hariri
* Graham Markall
* Guilherme Leobas
* Hannes Pahl
* Isuru Fernando
* Lucio Fernandez-Arjona
* Michael Collison
* Nick Sutcliffe
* Nick White
* Ollin Boer Bohan
* Sergey Pokhodenko
* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)
* Teugea Ioan-Teodor
* Thibault Ballier
* Tim Gates
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)
* ``@caljrobe``
* ``@imba-tjd``


Version 0.52.0 (30 November, 2020)
----------------------------------

This release focuses on performance improvements, but also adds some new
features and contains numerous bug fixes and stability improvements.

Highlights of core performance improvements include:

* Intel kindly sponsored research and development into producing a new reference
  count pruning pass. This pass operates at the LLVM level and can prune a
  number of common reference counting patterns. This will improve performance
  for two primary reasons:

  * There will be less pressure on the atomic locks used to do the reference
    counting.
  * Removal of reference counting operations permits more inlining and the
    optimisation passes can in general do more with what is present.

  (Siu Kwan Lam).
* Intel also sponsored work to improve the performance of the
  ``numba.typed.List`` container, particularly in the case of ``__getitem__``
  and iteration (Stuart Archibald).
* Superword-level parallelism vectorization is now switched on and the
  optimisation pipeline has been lightly analysed and tuned so as to be able to
  vectorize more and more often (Stuart Archibald).

Highlights of core feature changes include:

* The ``inspect_cfg`` method on the JIT dispatcher object has been
  significantly enhanced and now includes highlighted output and interleaved
  line markers and Python source (Stuart Archibald).
* The BSD operating system is now unofficially supported (Stuart Archibald).
* Numerous features/functionality improvements to NumPy support, including
  support for:

  * ``np.asfarray`` (Guilherme Leobas)
  * "subtyping" in record arrays (Lucio Fernandez-Arjona)
  * ``np.split`` and ``np.array_split`` (Isaac Virshup)
  * ``operator.contains`` with ``ndarray`` (``@mugoh``).
  * ``np.asarray_chkfinite`` (Rishabh Varshney).
  * NumPy 1.19 (Stuart Archibald).
  * the ``ndarray`` allocators, ``empty``, ``ones`` and ``zeros``, accepting a
    ``dtype`` specified as a string literal (Stuart Archibald).

* Booleans are now supported as literal types (Alexey Kozlov).
* On the CUDA target:

  * CUDA 9.0 is now the minimum supported version (Graham Markall).
  * Support for Unified Memory has been added (Max Katz).
  * Kernel launch overhead is reduced (Graham Markall).
  * Cudasim support for mapped array, memcopies and memset has been added (Mike
    Williams).
  * Access has been wired in to all libdevice functions (Graham Markall).
  * Additional CUDA atomic operations have been added (Michael Collison).
  * Additional math library functions (``frexp``, ``ldexp``, ``isfinite``)
    (Zhihao Yuan).
  * Support for ``power`` on complex numbers (Graham Markall).

Deprecations to note:

There are no new deprecations. However, note that "compatibility" mode, which
was added some 40 releases ago to help transition from 0.11 to 0.12+, has been
removed! Also, the shim to permit the import of ``jitclass`` from Numba's top
level namespace has now been removed as per the deprecation schedule.

General Enhancements:

* PR #5418: Add np.asfarray impl (Guilherme Leobas)
* PR #5560: Record subtyping (Lucio Fernandez-Arjona)
* PR #5609: Jitclass Infer Spec from Type Annotations (Ethan Pronovost)
* PR #5699: Implement np.split and np.array_split (Isaac Virshup)
* PR #6015: Adding BooleanLiteral type (Alexey Kozlov)
* PR #6027: Support operators inlining in InlineOverloads (Alexey Kozlov)
* PR #6038: Closes #6037, fixing FreeBSD compilation (László Károlyi)
* PR #6086: Add more accessible version information (Stuart Archibald)
* PR #6157: Add pipeline_class argument to @cfunc as supported by @jit. (Arthur
  Peters)
* PR #6262: Support dtype from str literal. (Stuart Archibald)
* PR #6271: Support ``ndarray`` contains (``@mugoh``)
* PR #6295: Enhance inspect_cfg (Stuart Archibald)
* PR #6304: Support NumPy 1.19 (Stuart Archibald)
* PR #6309: Add suitable file search path for BSDs. (Stuart Archibald)
* PR #6341: Re roll 6279 (Rishabh Varshney and Valentin Haenel)

Performance Enhancements:

* PR #6145: Patch to fingerprint namedtuples. (Stuart Archibald)
* PR #6202: Speed up str(int) (Stuart Archibald)
* PR #6261: Add np.ndarray.ptp() support. (Stuart Archibald)
* PR #6266: Use custom LLVM refcount pruning pass (Siu Kwan Lam)
* PR #6275: Switch on SLP vectorize. (Stuart Archibald)
* PR #6278: Improve typed list performance. (Stuart Archibald)
* PR #6335: Split optimisation passes. (Stuart Archibald)
* PR #6455: Fix refprune on obfuscated refs and stabilize optimisation WRT
  wrappers. (Stuart Archibald)

Fixes:

* PR #5639: Make UnicodeType inherit from Hashable (Stuart Archibald)
* PR #6006: Resolves incorrectly hoisted list in parfor. (Todd A. Anderson)
* PR #6126: fix version_info if version can not be determined (Valentin Haenel)
* PR #6137: Remove references to Python 2's long (Eric Wieser)
* PR #6139: Use direct syntax instead of the ``add_metaclass`` decorator (Eric
  Wieser)
* PR #6140: Replace calls to utils.iteritems(d) with d.items() (Eric Wieser)
* PR #6141: Fix #6130 objmode cache segfault (Siu Kwan Lam)
* PR #6156: Remove callers of ``reraise`` in favor of using ``with_traceback``
  directly (Eric Wieser)
* PR #6162: Move charseq support out of init (Stuart Archibald)
* PR #6165: #5425 continued (Amos Bird and Stuart Archibald)
* PR #6166: Remove Python 2 compatibility from numba.core.utils (Eric Wieser)
* PR #6185: Better error message on NotDefinedError (Luiz Almeida)
* PR #6194: Remove recursion from traverse_types (Radu Popovici)
* PR #6200: Workaround #5973 (Stuart Archibald)
* PR #6203: Make find_callname only lookup functions that are likely part of
  NumPy. (Stuart Archibald)
* PR #6204: Fix unicode kind selection for getitem. (Stuart Archibald)
* PR #6206: Build all extension modules with -g -Wall -Werror on Linux x86,
  provide -O0 flag option (Graham Markall)
* PR #6212: Fix for objmode recompilation issue (Alexey Kozlov)
* PR #6213: Fix #6177. Remove AOT dependency on the Numba package (Siu Kwan Lam)
* PR #6224: Add support for tuple concatenation to array analysis. (#5396
  continued) (Todd A. Anderson)
* PR #6231: Remove compatibility mode (Graham Markall)
* PR #6254: Fix win-32 hashing bug (from Stuart Archibald) (Ray Donnelly)
* PR #6265: Fix #6260 (Stuart Archibald)
* PR #6267: speed up a couple of really slow unittests (Stuart Archibald)
* PR #6281: Remove numba.jitclass shim as per deprecation schedule. (Stuart
  Archibald)
* PR #6294: Make return type propagate to all return variables (Andreas Sodeur)
* PR #6300: Un-skip tests that were skipped because of #4026. (Owen Anderson)
* PR #6307: Remove restrictions on SVML version due to bug in LLVM SVML CC
  (Stuart Archibald)
* PR #6316: Make IR inliner tests not self mutating. (Stuart Archibald)
* PR #6318: PR #5892 continued (Todd A. Anderson, via Stuart Archibald)
* PR #6319: Permit switching off boundschecking when debug is on. (Stuart
  Archibald)
* PR #6324: PR 6208 continued (Ivan Butygin and Stuart Archibald)
* PR #6337: Implements ``key`` on ``types.TypeRef`` (Andreas Sodeur)
* PR #6354: Bump llvmlite to 0.35. series. (Stuart Archibald)
* PR #6357: Fix enumerate invalid decref (Siu Kwan Lam)
* PR #6359: Fixes typed list indexing on 32bit (Stuart Archibald)
* PR #6378: Fix incorrect CPU override in vectorization test. (Stuart Archibald)
* PR #6379: Use O0 to enable inline and not affect loop-vectorization by later
  O3... (Siu Kwan Lam)
* PR #6384: Fix failing tests to match on platform invariant int spelling.
  (Stuart Archibald)
* PR #6390: Updates inspect_cfg (Stuart Archibald)
* PR #6396: Remove hard dependency on tbb package. (Stuart Archibald)
* PR #6408: Don't do array analysis for tuples that contain arrays. (Todd A.
  Anderson)
* PR #6441: Fix ASCII flag in Unicode slicing (0.52.0rc2 regression) (Ehsan
  Totoni)
* PR #6442: Fix array analysis regression in 0.52 RC2 for tuple of 1D arrays
  (Ehsan Totoni)
* PR #6446: Fix #6444: pruner issues with reference stealing functions (Siu
  Kwan Lam)
* PR #6450: Fix asfarray kwarg default handling. (Stuart Archibald)
* PR #6486: fix abstract base class import (Valentin Haenel)
* PR #6487: Restrict maximum version of python (Siu Kwan Lam)
* PR #6527: setup.py: fix py version guard (Chris Barnes)

CUDA Enhancements/Fixes:

* PR #5465: Remove macro expansion and replace uses with FE typing + BE lowering
  (Graham Markall)
* PR #5741: CUDA: Add two-argument implementation of round() (Graham Markall)
* PR #5900: Enable CUDA Unified Memory (Max Katz)
* PR #6042: CUDA: Lower launch overhead by launching kernel directly (Graham
  Markall)
* PR #6064: Lower math.frexp and math.ldexp in numba.cuda (Zhihao Yuan)
* PR #6066: Lower math.isfinite in numba.cuda (Zhihao Yuan)
* PR #6092: CUDA: Add mapped_array_like and pinned_array_like (Graham Markall)
* PR #6127: Fix race in reduction kernels on Volta, require CUDA 9, add syncwarp
  with default mask (Graham Markall)
* PR #6129: Extend Cudasim to support most of the memory functionality. (Mike
  Williams)
* PR #6150: CUDA: Turn on flake8 for cudadrv and fix errors (Graham Markall)
* PR #6152: CUDA: Provide wrappers for all libdevice functions, and fix typing
  of math function (#4618) (Graham Markall)
* PR #6227: Raise exception when no supported architectures are found (Jacob
  Tomlinson)
* PR #6244: CUDA Docs: Make workflow using simulator more explicit (Graham
  Markall)
* PR #6248: Add support for CUDA atomic subtract operations (Michael Collison)
* PR #6289: Refactor atomic test cases to reduce code duplication (Michael
  Collison)
* PR #6290: CUDA: Add support for complex power (Graham Markall)
* PR #6296: Fix flake8 violations in numba.cuda module (Graham Markall)
* PR #6297: Fix flake8 violations in numba.cuda.tests.cudapy module (Graham
  Markall)
* PR #6298: Fix flake8 violations in numba.cuda.tests.cudadrv (Graham Markall)
* PR #6299: Fix flake8 violations in numba.cuda.simulator (Graham Markall)
* PR #6306: Fix flake8 in cuda atomic test from merge. (Stuart Archibald)
* PR #6325: Refactor code for atomic operations (Michael Collison)
* PR #6329: Flake8 fix for a CUDA test (Stuart Archibald)
* PR #6331: Explicitly state that NUMBA_ENABLE_CUDASIM needs to be set before
  import (Graham Markall)
* PR #6340: CUDA: Fix #6339, performance regression launching specialized
  kernels (Graham Markall)
* PR #6380: Only test managed allocations on Linux (Graham Markall)

Documentation Updates:

* PR #6090: doc: Add doc on direct creation of Numba typed-list (``@rht``)
* PR #6110: Update CONTRIBUTING.md (Stuart Archibald)
* PR #6128: CUDA Docs: Restore Dispatcher.forall() docs (Graham Markall)
* PR #6277: fix: cross2d wrong doc. reference (issue #6276) (``@jeertmans``)
* PR #6282: Remove docs on Python 2(.7) EOL. (Stuart Archibald)
* PR #6283: Add note on how public CI is impl and what users can do to help.
  (Stuart Archibald)
* PR #6292: Document support for structured array attribute access
  (Graham Markall)
* PR #6310: Declare unofficial \*BSD support (Stuart Archibald)
* PR #6342: Fix docs on literally usage. (Stuart Archibald)
* PR #6348: doc: fix typo in jitclass.rst ("initilising" -> "initialising")
  (``@muxator``)
* PR #6362: Move llvmlite support in README to 0.35 (Stuart Archibald)
* PR #6363: Note that reference counted types are not permitted in set().
  (Stuart Archibald)
* PR #6364: Move deprecation schedules for 0.52 (Stuart Archibald)

CI/Infrastructure Updates:

* PR #6252: Show channel URLs (Siu Kwan Lam)
* PR #6338: Direct user questions to Discourse instead of the Google Group.
  (Stan Seibert)
* PR #6474: Add skip on PPC64LE for tests causing SIGABRT in LLVM. (Stuart
  Archibald)

Authors:

* Alexey Kozlov
* Amos Bird
* Andreas Sodeur
* Arthur Peters
* Chris Barnes
* Ehsan Totoni (core dev)
* Eric Wieser
* Ethan Pronovost
* Graham Markall
* Guilherme Leobas
* Isaac Virshup
* Ivan Butygin
* Jacob Tomlinson
* Luiz Almeida
* László Károlyi
* Lucio Fernandez-Arjona
* Max Katz
* Michael Collison
* Mike Williams
* Owen Anderson
* Radu Popovici
* Ray Donnelly
* Rishabh Varshney
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)
* Zhihao Yuan
* ``@jeertmans``
* ``@mugoh``
* ``@muxator``
* ``@rht``



Version 0.51.2 (September 2, 2020)
----------------------------------

This is a bugfix release for 0.51.1. It fixes a critical performance bug in the
CFG back edge computation algorithm that leads to exponential time complexity
arising in compilation for use cases with certain pathological properties.

* PR #6195: PR 6187 Continue. Don't visit already checked successors

Authors:

* Graham Markall
* Siu Kwan Lam (core dev)


Version 0.51.1 (August 26, 2020)
--------------------------------

This is a bugfix release for 0.51.0, it fixes a critical bug in caching, another
critical bug in the CUDA target initialisation sequence and also fixes some
compile time performance regressions:

* PR #6141: Fix #6130 objmode cache segfault
* PR #6146: Fix compilation slowdown due to controlflow analysis
* PR #6147: CUDA: Don't make a runtime call on import
* PR #6153: Fix for #6151. Make UnicodeCharSeq into str for comparison.
* PR #6168: Fix Issue #6167: Failure in test_cuda_submodules

Authors:

* Graham Markall
* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)


Version 0.51.0 (August 12, 2020)
--------------------------------

This release continues to add new features to Numba and also contains a
significant number of bug fixes and stability improvements.

Highlights of core feature changes include:

* The compilation chain is now based on LLVM 10 (Valentin Haenel).
* Numba has internally switched to prefer non-literal types over literal ones so
  as to reduce function over-specialisation, this with view of speeding up
  compile times (Siu Kwan Lam).
* On the CUDA target: Support for CUDA Toolkit 11, Ampere, and Compute
  Capability 8.0; Printing of ``SASS`` code for kernels; Callbacks to Python
  functions can be inserted into CUDA streams, and streams are async awaitable;
  Atomic ``nanmin`` and ``nanmax`` functions are added; Fixes for various
  miscompilations and segfaults. (mostly Graham Markall; call backs on
  streams by Peter Würtz).

Intel also kindly sponsored research and development that lead to some exciting
new features:

* Support for heterogeneous immutable lists and heterogeneous immutable string
  key dictionaries. Also optional initial/construction value capturing for all
  lists and dictionaries containing literal values (Stuart Archibald).
* A new pass-by-reference mutable structure extension type ``StructRef`` (Siu
  Kwan Lam).
* Object mode blocks are now cacheable, with the side effect of numerous bug
  fixes and performance improvements in caching. This also permits caching of
  functions defined in closures (Siu Kwan Lam).

Deprecations to note:

To align with other targets, the ``argtypes`` and ``restypes`` kwargs to
``@cuda.jit`` are now deprecated, the ``bind`` kwarg is also deprecated.
Further the ``target`` kwarg to the ``numba.jit`` decorator family is
deprecated.

General Enhancements:

* PR #5463: Add str(int) impl
* PR #5526: Impl. np.asarray(literal)
* PR #5619: Add support for multi-output ufuncs
* PR #5711: Division with timedelta input
* PR #5763: Support minlength argument to np.bincount
* PR #5779: Return zero array from np.dot when the arguments are empty.
* PR #5796: Add implementation for np.positive
* PR #5849: Setitem for records when index is StringLiteral, including literal
  unroll
* PR #5856: Add support for conversion of inplace_binop to parfor.
* PR #5893: Allocate 1D iteration space one at a time for more even
  distribution.
* PR #5922: Reduce objmode and unpickling overhead
* PR #5944: re-enable OpenMP in wheels
* PR #5946: Implement literal dictionaries and lists.
* PR #5956: Update numba_sysinfo.py
* PR #5978: Add structref as a mutable struct that is pass-by-ref
* PR #5980: Deprecate target kwarg for numba.jit.
* PR #6058: Add prefer_literal option to overload API

Fixes:

* PR #5674: Fix #3955. Allow `with objmode` to be cached
* PR #5724: Initialize process lock lazily to prevent multiprocessing issue
* PR #5783: Make np.divide and np.remainder code more similar
* PR #5808: Fix 5665 Block jit(nopython=True, forceobj=True) and suppress
  njit(forceobj=True)
* PR #5834: Fix the is operator on Ellipsis
* PR #5838: Ensure ``Dispatcher.__eq__`` always returns a bool
* PR #5841: cleanup: Use PythonAPI.bool_from_bool in more places
* PR #5862: Do not leak loop iteration variables into the numba.np.npyimpl
  namespace
* PR #5869: Update repomap
* PR #5879: Fix erroneous input mutation in linalg routines
* PR #5882: Type check function in jit decorator
* PR #5925: Use np.inf and -np.inf for max and min float values respectively.
* PR #5935: Fix default arguments with multiprocessing
* PR #5952: Fix "Internal error ... local variable 'errstr' referenced before
  assignment during BoundFunction(...)"
* PR #5962: Fix SVML tests with LLVM 10 and AVX512
* PR #5972: fix flake8 for numba/runtests.py
* PR #5995: Update setup.py with new llvmlite versions
* PR #5996: Set lower bound for llvmlite to 0.33
* PR #6004: Fix problem in branch pruning with LiteralStrKeyDict
* PR #6017: Fixing up numba_do_raise
* PR #6028: Fix #6023
* PR #6031: Continue 5821
* PR #6035: Fix overspecialize of literal
* PR #6046: Fixes statement reordering bug in maximize fusion step.
* PR #6056: Fix issue on invalid inlining of non-empty build_list by
  inline_arraycall
* PR #6057: fix aarch64/python_3.8 failure on master
* PR #6070: Fix overspecialized containers
* PR #6071: Remove f-strings in setup.py
* PR #6072: Fix for #6005
* PR #6073: Fixes invalid C prototype in helper function.
* PR #6078: Duplicate NumPy's PyArray_DescrCheck macro
* PR #6081: Fix issue with cross drive use and relpath.
* PR #6083: Fix bug in initial value unify.
* PR #6087: remove invalid sanity check from randrange tests
* PR #6089: Fix invalid reference to TypingError
* PR #6097: Add function code and closure bytes into cache key
* PR #6099: Restrict upper limit of TBB version due to ABI changes.
* PR #6101: Restrict lower limit of icc_rt version due to assumed SVML bug.
* PR #6107: Fix and test #6095
* PR #6109: Fixes an issue reported in #6094
* PR #6111: Decouple LiteralList and LiteralStrKeyDict from tuple
* PR #6116: Fix #6102. Problem with non-unique label.

CUDA Enhancements/Fixes:

* PR #5359: Remove special-casing of 0d arrays
* PR #5709: CUDA: Refactoring of cuda.jit and kernel / dispatcher abstractions
* PR #5732: CUDA Docs: document ``forall`` method of kernels
* PR #5745: CUDA stream callbacks and async awaitable streams
* PR #5761: Add implmentation for int types for isnan and isinf for CUDA
* PR #5819: Add support for CUDA 11 and Ampere / CC 8.0
* PR #5826: CUDA: Add function to get SASS for kernels
* PR #5846: CUDA: Allow disabling NVVM optimizations, and fix debug issues
* PR #5851: CUDA EMM enhancements - add default get_ipc_handle implementation,
  skip a test conditionally
* PR #5852: CUDA: Fix ``cuda.test()``
* PR #5857: CUDA docs: Add notes on resetting the EMM plugin
* PR #5859: CUDA: Fix reduce docs and style improvements
* PR #6016: Fixes change of list spelling in a cuda test.
* PR #6020: CUDA: Fix #5820, adding atomic nanmin / nanmax
* PR #6030: CUDA: Don't optimize IR before sending it to NVVM
* PR #6052: Fix dtype for atomic_add_double testsuite
* PR #6080: CUDA: Prevent auto-upgrade of atomic intrinsics
* PR #6123: Fix #6121

Documentation Updates:

* PR #5782: Host docs on Read the Docs
* PR #5830: doc: Mention that caching uses pickle
* PR #5963: Fix broken link to numpy ufunc signature docs
* PR #5975: restructure communication section
* PR #5981: Document bounds-checking behavior in python deviations page
* PR #5993: Docs for structref
* PR #6008: Small fix so bullet points are rendered by sphinx
* PR #6013: emphasize cuda kernel functions are asynchronous
* PR #6036: Update deprecation doc from numba.errors to numba.core.errors
* PR #6062: Change references to numba.pydata.org to https

CI updates:

* PR #5850: Updates the "New Issue" behaviour to better redirect users.
* PR #5940: Add discourse badge
* PR #5960: Setting mypy on CI

Enhancements from user contributed PRs (with thanks!):

* Aisha Tammy added the ability to switch off TBB support at compile time in
  #5821 (continued in #6031 by Stuart Archibald).
* Alexander Stiebing fixed a reference before assignment bug in #5952.
* Alexey Kozlov fixed a bug in tuple getitem for literals in #6028.
* Andrew Eckart updated the repomap in #5869, added support for Read the Docs
  in #5782, fixed a bug in the ``np.dot`` implementation to correctly handle
  empty arrays in #5779 and added support for ``minlength`` to ``np.bincount``
  in #5763.
* ``@bitsisbits`` updated ``numba_sysinfo.py`` to handle HSA agents correctly in
  #5956.
* Daichi Suzuo Fixed a bug in the threading backend initialisation sequence such
  that it is now correctly a lazy lock in #5724.
* Eric Wieser contributed a number of patches, particularly in enhancing and
  improving the ``ufunc`` capabilities:

  * #5359: Remove special-casing of 0d arrays
  * #5834: Fix the is operator on Ellipsis
  * #5619: Add support for multi-output ufuncs
  * #5841: cleanup: Use PythonAPI.bool_from_bool in more places
  * #5862: Do not leak loop iteration variables into the numba.np.npyimpl
    namespace
  * #5838: Ensure ``Dispatcher.__eq__`` always returns a bool
  * #5830: doc: Mention that caching uses pickle
  * #5783: Make np.divide and np.remainder code more similar

* Ethan Pronovost added a guard to prevent the common mistake of applying a jit
  decorator to the same function twice in #5881.
* Graham Markall contributed many patches to the CUDA target, as follows:

  * #6052: Fix dtype for atomic_add_double tests
  * #6030: CUDA: Don't optimize IR before sending it to NVVM
  * #5846: CUDA: Allow disabling NVVM optimizations, and fix debug issues
  * #5826: CUDA: Add function to get SASS for kernels
  * #5851: CUDA EMM enhancements - add default get_ipc_handle implementation,
    skip a test conditionally
  * #5709: CUDA: Refactoring of cuda.jit and kernel / dispatcher abstractions
  * #5819: Add support for CUDA 11 and Ampere / CC 8.0
  * #6020: CUDA: Fix #5820, adding atomic nanmin / nanmax
  * #5857: CUDA docs: Add notes on resetting the EMM plugin
  * #5859: CUDA: Fix reduce docs and style improvements
  * #5852: CUDA: Fix ``cuda.test()``
  * #5732: CUDA Docs: document ``forall`` method of kernels

* Guilherme Leobas added support for ``str(int)`` in #5463 and
  ``np.asarray(literal value)``` in #5526.
* Hameer Abbasi deprecated the ``target`` kwarg for ``numba.jit`` in #5980.
* Hannes Pahl added a badge to the Numba github page linking to the new
  discourse forum in #5940 and also fixed a bug that permitted illegal
  combinations of flags to be passed into ``@jit`` in #5808.
* Kayran Schmidt emphasized that CUDA kernel functions are asynchronous in the
  documentation in #6013.
* Leonardo Uieda fixed a broken link to the NumPy ufunc signature docs in #5963.
* Lucio Fernandez-Arjona added mypy to CI and started adding type annotations to
  the code base in #5960, also fixed a (de)serialization problem on the
  dispatcher in #5935, improved the undefined variable error message in #5876,
  added support for division with timedelta input in #5711 and implemented
  ``setitem`` for records when the index is a ``StringLiteral`` in #5849.
* Ludovic Tiako documented Numba's bounds-checking behavior in the python
  deviations page in #5981.
* Matt Roeschke changed all ``http`` references ``https`` in #6062.
* ``@niteya-shah`` implemented ``isnan`` and ``isinf`` for integer types on the
  CUDA target in #5761 and implemented ``np.positive`` in #5796.
* Peter Würtz added CUDA stream callbacks and async awaitable streams in #5745.
* ``@rht`` fixed an invalid import referred to in the deprecation documentation
  in #6036.
* Sergey Pokhodenko updated the SVML tests for LLVM 10 in #5962.
* Shyam Saladi fixed a Sphinx rendering bug in #6008.

Authors:

* Aisha Tammy
* Alexander Stiebing
* Alexey Kozlov
* Andrew Eckart
* ``@bitsisbits``
* Daichi Suzuo
* Eric Wieser
* Ethan Pronovost
* Graham Markall
* Guilherme Leobas
* Hameer Abbasi
* Hannes Pahl
* Kayran Schmidt
* Kozlov, Alexey
* Leonardo Uieda
* Lucio Fernandez-Arjona
* Ludovic Tiako
* Matt Roeschke
* ``@niteya-shah``
* Peter Würtz
* Sergey Pokhodenko
* Shyam Saladi
* ``@rht``
* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)


Version 0.50.1 (Jun 24, 2020)
-----------------------------

This is a bugfix release for 0.50.0, it fixes a critical bug in error reporting
and a number of other smaller issues:

* PR #5861: Added except for possible Windows get_terminal_size exception
* PR #5876: Improve undefined variable error message
* PR #5884: Update the deprecation notices for 0.50.1
* PR #5889: Fixes literally not forcing re-dispatch for inline='always'
* PR #5912: Fix bad attr access on certain typing templates breaking exceptions.
* PR #5918: Fix cuda test due to #5876

Authors:

* ``@pepping_dore``
* Lucio Fernandez-Arjona
* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)


Version 0.50.0 (Jun 10, 2020)
-----------------------------

This is a more usual release in comparison to the others that have been made in
the last six months. It comprises the result of a number of maintenance tasks
along with some new features and a lot of bug fixes.

Highlights of core feature changes include:

* The compilation chain is now based on LLVM 9.
* The error handling and reporting system has been improved to reduce the size
  of error messages, and also improve quality and specificity.
* The CUDA target has more stream constructors available and a new function for
  compiling to PTX without linking and loading the code to a device. Further,
  the macro-based system for describing CUDA threads and blocks has been
  replaced with standard typing and lowering implementations, for improved
  debugging and extensibility.

IMPORTANT: The backwards compatibility shim, that was present in 0.49.x to
accommodate the refactoring of Numba's internals, has been removed. If a module
is imported from a moved location an ``ImportError`` will occur.

General Enhancements:

* PR #5060: Enables np.sum for timedelta64
* PR #5225: Adjust interpreter to make conditionals predicates via bool() call.
* PR #5506: Jitclass static methods
* PR #5580: Revert shim
* PR #5591: Fix #5525 Add figure for total memory to ``numba -s`` output.
* PR #5616: Simplify the ufunc kernel registration
* PR #5617: Remove /examples from the Numba repo.
* PR #5673: Fix inliners to run all passes on IR and clean up correctly.
* PR #5700: Make it easier to understand type inference: add SSA dump, use for
  ``DEBUG_TYPEINFER``
* PR #5702: Fixes for LLVM 9
* PR #5722: Improve error messages.
* PR #5758: Support NumPy 1.18

Fixes:

* PR #5390: add error handling for lookup_module
* PR #5464: Jitclass drops annotations to avoid error
* PR #5478: Fix #5471. Issue with omitted type not recognized as literal value.
* PR #5517: Fix numba.typed.List extend for singleton and empty iterable
* PR #5549: Check type getitem
* PR #5568: Add skip to entrypoint test on windows
* PR #5581: Revert #5568
* PR #5602: Fix segfault caused by pop from numba.typed.List
* PR #5645: Fix SSA redundant CFG computation
* PR #5686: Fix issue with SSA not minimal
* PR #5689: Fix bug in unified_function_type (issue 5685)
* PR #5694: Skip part of slice array analysis if any part is not analyzable.
* PR #5697: Fix usedef issue with parfor loopnest variables.
* PR #5705: A fix for cases where SSA looks like a reduction variable.
* PR #5714: Fix bug in test
* PR #5717: Initialise Numba extensions ahead of any compilation starting.
* PR #5721: Fix array iterator layout.
* PR #5738: Unbreak master on buildfarm
* PR #5757: Force LLVM to use ZMM registers for vectorization.
* PR #5764: fix flake8 errors
* PR #5768: Interval example: fix import
* PR #5781: Moving record array examples to a test module
* PR #5791: Fix up no cgroups problem
* PR #5795: Restore refct removal pass and make it strict
* PR #5807: Skip failing test on POWER8 due to PPC CTR Loop problem.
* PR #5812: Fix side issue from #5792, @overload inliner cached IR being
  mutated.
* PR #5815: Pin llvmlite to 0.33
* PR #5833: Fixes the source location appearing incorrectly in error messages.

CUDA Enhancements/Fixes:

* PR #5347: CUDA: Provide more stream constructors
* PR #5388: CUDA: Fix OOB write in test_round{f4,f8}
* PR #5437: Fix #5429: Exception using ``.get_ipc_handle(...)`` on array from
  ``as_cuda_array(...)``
* PR #5481: CUDA: Replace macros with typing and lowering implementations
* PR #5556: CUDA: Make atomic semantics match Python / NumPy, and fix #5458
* PR #5558: CUDA: Only release primary ctx if retained
* PR #5561: CUDA: Add function for compiling to PTX (+ other small fixes)
* PR #5573: CUDA: Skip tests under cuda-memcheck that hang it
* PR #5578: Implement math.modf for CUDA target
* PR #5704: CUDA Eager compilation: Fix max_registers kwarg
* PR #5718: CUDA lib path tests: unset CUDA_PATH when CUDA_HOME unset
* PR #5800: Fix LLVM 9 IR for NVVM
* PR #5803: CUDA Update expected error messages to fix #5797

Documentation Updates:

* PR #5546: DOC: Add documentation about cost model to inlining notes.
* PR #5653: Update doc with respect to try-finally case

Enhancements from user contributed PRs (with thanks!):

* Elias Kuthe fixed in issue with imports in the Interval example in #5768
* Eric Wieser Simplified the ufunc kernel registration mechanism in #5616
* Ethan Pronovost patched a problem with ``__annotations__`` in ``jitclass`` in
  #5464, fixed a bug that lead to infinite loops in Numba's ``Type.__getitem__``
  in #5549, fixed a bug in ``np.arange`` testing in #5714 and added support for
  ``@staticmethod`` to ``jitclass`` in #5506.
* Gabriele Gemmi implemented ``math.modf`` for the CUDA target in #5578
* Graham Markall contributed many patches, largely to the CUDA target, as
  follows:

  * #5347: CUDA: Provide more stream constructors
  * #5388: CUDA: Fix OOB write in test_round{f4,f8}
  * #5437: Fix #5429: Exception using ``.get_ipc_handle(...)`` on array from
    ``as_cuda_array(...)``
  * #5481: CUDA: Replace macros with typing and lowering implementations
  * #5556: CUDA: Make atomic semantics match Python / NumPy, and fix #5458
  * #5558: CUDA: Only release primary ctx if retained
  * #5561: CUDA: Add function for compiling to PTX (+ other small fixes)
  * #5573: CUDA: Skip tests under cuda-memcheck that hang it
  * #5648: Unset the memory manager after EMM Plugin tests
  * #5700: Make it easier to understand type inference: add SSA dump, use for
    ``DEBUG_TYPEINFER``
  * #5704: CUDA Eager compilation: Fix max_registers kwarg
  * #5718: CUDA lib path tests: unset CUDA_PATH when CUDA_HOME unset
  * #5800: Fix LLVM 9 IR for NVVM
  * #5803: CUDA Update expected error messages to fix #5797

* Guilherme Leobas updated the documentation surrounding try-finally in #5653
* Hameer Abbasi added documentation about the cost model to the notes on
  inlining in #5546
* Jacques Gaudin rewrote ``numba -s`` to produce and consume a dictionary of
  output about the current system in #5591
* James Bourbeau Updated min/argmin and max/argmax to handle non-leading nans
  (via #5758)
* Lucio Fernandez-Arjona moved the record array examples to a test module in
  #5781 and added ``np.timedelta64`` handling to ``np.sum`` in #5060
* Pearu Peterson Fixed a bug in unified_function_type in #5689
* Sergey Pokhodenko fixed an issue impacting LLVM 10 regarding vectorization
  widths on Intel SkyLake processors in #5757
* Shan Sikdar added error handling for ``lookup_module`` in #5390
* @toddrme2178 add CI testing for NumPy 1.18 (via #5758)

Authors:

* Elias Kuthe
* Eric Wieser
* Ethan Pronovost
* Gabriele Gemmi
* Graham Markall
* Guilherme Leobas
* Hameer Abbasi
* Jacques Gaudin
* James Bourbeau
* Lucio Fernandez-Arjona
* Pearu Peterson
* Sergey Pokhodenko
* Shan Sikdar
* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* ``@toddrme2178``
* Valentin Haenel (core dev)


Version 0.49.1 (May 7, 2020)
----------------------------

This is a bugfix release for 0.49.0, it fixes some residual issues with SSA
form, a critical bug in the branch pruning logic and a number of other smaller
issues:

* PR #5587: Fixed #5586 Threading Implementation Typos
* PR #5592: Fixes #5583 Remove references to cffi_support from docs and examples
* PR #5614: Fix invalid type in resolve for comparison expr in parfors.
* PR #5624: Fix erroneous rewrite of predicate to bit const on prune.
* PR #5627: Fixes #5623, SSA local def scan based on invalid equality
  assumption.
* PR #5629: Fixes naming error in array_exprs
* PR #5630: Fix #5570. Incorrect race variable detection due to SSA naming.
* PR #5638: Make literal_unroll function work as a freevar.
* PR #5648: Unset the memory manager after EMM Plugin tests
* PR #5651: Fix some SSA issues
* PR #5652: Pin to sphinx=2.4.4 to avoid problem with C declaration
* PR #5658: Fix unifying undefined first class function types issue
* PR #5669: Update example in 5m guide WRT SSA type stability.
* PR #5676: Restore ``numba.types`` as public API

Authors:

* Graham Markall
* Juan Manuel Cruz Martinez
* Pearu Peterson
* Sean Law
* Stuart Archibald (core dev)
* Siu Kwan Lam (core dev)


Version 0.49.0 (Apr 16, 2020)
-----------------------------

This release is very large in terms of code changes. Large scale removal of
unsupported Python and NumPy versions has taken place along with a significant
amount of refactoring to simplify the Numba code base to make it easier for
contributors. Numba's intermediate representation has also undergone some
important changes to solve a number of long standing issues. In addition some
new features have been added and a large number of bugs have been fixed!

IMPORTANT: In this release Numba's internals have moved about a lot. A backwards
compatibility "shim" is provided for this release so as to not immediately break
projects using Numba's internals. If a module is imported from a moved location
the shim will issue a deprecation warning and suggest how to update the import
statement for the new location. The shim will be removed in 0.50.0!

Highlights of core feature changes include:

* Removal of all Python 2 related code and also updating the minimum supported
  Python version to 3.6, the minimum supported NumPy version to 1.15 and the
  minimum supported SciPy version to 1.0. (Stuart Archibald).
* Refactoring of the Numba code base. The code is now organised into submodules
  by functionality. This cleans up Numba's top level namespace.
  (Stuart Archibald).
* Introduction of an ``ir.Del`` free static single assignment form for Numba's
  intermediate representation (Siu Kwan Lam and Stuart Archibald).
* An OpenMP-like thread masking API has been added for use with code using the
  parallel CPU backends (Aaron Meurer and Stuart Archibald).
* For the CUDA target, all kernel launches now require a configuration, this
  preventing accidental launches of kernels with the old default of a single
  thread in a single block. The hard-coded autotuner is also now removed, such
  tuning is deferred to CUDA API calls that provide the same functionality
  (Graham Markall).
* The CUDA target also gained an External Memory Management plugin interface to
  allow Numba to use another CUDA-aware library for all memory allocations and
  deallocations (Graham Markall).
* The Numba Typed List container gained support for construction from iterables
  (Valentin Haenel).
* Experimental support was added for first-class function types
  (Pearu Peterson).

Enhancements from user contributed PRs (with thanks!):

* Aaron Meurer added support for thread masking at runtime in #4615.
* Andreas Sodeur fixed a long standing bug that was preventing ``cProfile`` from
  working with Numba JIT compiled functions in #4476.
* Arik Funke fixed error messages in ``test_array_reductions`` (#5278), fixed
  an issue with test discovery (#5239), made it so the documentation would build
  again on windows (#5453) and fixed a nested list problem in the docs in #5489.
* Antonio Russo fixed a SyntaxWarning in #5252.
* Eric Wieser added support for inferring the types of object arrays (#5348) and
  iterating over 2D arrays (#5115), also fixed some compiler warnings due to
  missing (void) in #5222. Also helped improved the "shim" and associated
  warnings in #5485, #5488, #5498 and partly #5532.
* Ethan Pronovost fixed a problem with the shim erroneously warning for jitclass
  use in #5454 and also prevented illegal return values in jitclass ``__init__``
  in #5505.
* Gabriel Majeri added SciPy 2019 talks to the docs in #5106.
* Graham Markall changed the Numba HTML documentation theme to resolve a number
  of long standing issues in #5346. Also contributed were a large number of CUDA
  enhancements and fixes, namely:

  * #5519: CUDA: Silence the test suite - Fix #4809, remove autojit, delete
    prints
  * #5443: Fix #5196: Docs: assert in CUDA only enabled for debug
  * #5436: Fix #5408: test_set_registers_57 fails on Maxwell
  * #5423: Fix #5421: Add notes on printing in CUDA kernels
  * #5400: Fix #4954, and some other small CUDA testsuite fixes
  * #5328: NBEP 7: External Memory Management Plugin Interface
  * #5144: Fix #4875: Make #2655 test with debug expect to pass
  * #5323: Document lifetime semantics of CUDA Array Interface
  * #5061: Prevent kernel launch with no configuration, remove autotuner
  * #5099: Fix #5073: Slices of dynamic shared memory all alias
  * #5136: CUDA: Enable asynchronous operations on the default stream
  * #5085: Support other itemsizes with view
  * #5059: Docs: Explain how to use Memcheck with Numba, fixups in CUDA
    documentation
  * #4957: Add notes on overwriting gufunc inputs to docs

* Greg Jennings fixed an issue with ``np.random.choice`` not acknowledging the
  RNG seed correctly in #3897/#5310.
* Guilherme Leobas added support for ``np.isnat`` in #5293.
* Henry Schreiner made the llvmlite requirements more explicit in
  requirements.txt in #5150.
* Ivan Butygin helped fix an issue with parfors sequential lowering in
  #5114/#5250.
* Jacques Gaudin fixed a bug for Python >= 3.8 in ``numba -s`` in #5548.
* Jim Pivarski added some hints for debugging entry points in #5280.
* John Kirkham added ``numpy.dtype`` coercion for the ``dtype`` argument to CUDA
  device arrays in #5252.
* Leo Fang added a list of libraries that support ``__cuda_array_interface__``
  in #5104.
* Lucio Fernandez-Arjona added ``getitem`` for the NumPy record type when the
  index is a ``StringLiteral`` type in #5182 and improved the documentation
  rendering via additions to the TOC and removal of numbering in #5450.
* Mads R. B. Kristensen fixed an issue with ``__cuda_array_interface__`` not
  requiring the context in #5189.
* Marcin Tolysz added support for nested modules in AOT compilation in #5174.
* Mike Williams fixed some issues with NumPy records and ``getitem`` in the CUDA
  simulator in #5343.
* Pearu Peterson added experimental support for first-class function types in
  #5287 (and fixes in #5459, #5473/#5429, and #5557).
* Ravi Teja Gutta added support for ``np.flip`` in #4376/#5313.
* Rohit Sanjay fixed an issue with type refinement for unicode input supplied to
  typed-list ``extend()`` (#5295) and fixed unicode ``.strip()`` to strip all
  whitespace characters in #5213.
* Vladimir Lukyanov fixed an awkward bug in ``typed.dict`` in #5361, added a fix
  to ensure the LLVM and assembly dumps are highlighted correctly in #5357 and
  implemented a Numba IR Lexer and added highlighting to Numba IR dumps in
  #5333.
* hdf fixed an issue with the ``boundscheck`` flag in the CUDA jit target in
  #5257.

General Enhancements:

* PR #4615: Allow masking threads out at runtime
* PR #4798: Add branch pruning based on raw predicates.
* PR #5115: Add support for iterating over 2D arrays
* PR #5117: Implement ord()/chr()
* PR #5122: Remove Python 2.
* PR #5127: Calling convention adaptor for boxer/unboxer to call jitcode
* PR #5151: implement None-typed typed-list
* PR #5174: Nested modules https://github.com/numba/numba/issues/4739
* PR #5182: Add getitem for Record type when index is StringLiteral
* PR #5185: extract code-gen utilities from closures
* PR #5197: Refactor Numba, part I
* PR #5210: Remove more unsupported Python versions from build tooling.
* PR #5212: Adds support for viewing the CFG of the ELF disassembly.
* PR #5227: Immutable typed-list
* PR #5231: Added support for ``np.asarray`` to be used with
  ``numba.typed.List``
* PR #5235: Added property ``dtype`` to ``numba.typed.List``
* PR #5272: Refactor parfor: split up ParforPass
* PR #5281: Make IR ir.Del free until legalized.
* PR #5287: First-class function type
* PR #5293: np.isnat
* PR #5294: Create typed-list from iterable
* PR #5295: refine typed-list on unicode input to extend
* PR #5296: Refactor parfor: better exception from passes
* PR #5308: Provide ``numba.extending.is_jitted``
* PR #5320: refactor array_analysis
* PR #5325: Let literal_unroll accept types.Named*Tuple
* PR #5330: refactor common operation in parfor lowering into a new util
* PR #5333: Add: highlight Numba IR dump
* PR #5342: Support for tuples passed to parfors.
* PR #5348: Add support for inferring the types of object arrays
* PR #5351: SSA again
* PR #5352: Add shim to accommodate refactoring.
* PR #5356: implement allocated parameter in njit
* PR #5369: Make test ordering more consistent across feature availability
* PR #5428: Wip/deprecate jitclass location
* PR #5441: Additional changes to first class function
* PR #5455: Move to llvmlite 0.32.*
* PR #5457: implement repr for untyped lists

Fixes:

* PR #4476: Another attempt at fixing frame injection in the dispatcher tracing
  path
* PR #4942: Prevent some parfor aliasing.  Rename copied function var to prevent
  recursive type locking.
* PR #5092: Fix 5087
* PR #5150: More explicit llvmlite requirement in requirements.txt
* PR #5172: fix version spec for llvmlite
* PR #5176: Normalize kws going into fold_arguments.
* PR #5183: pass 'inline' explicitly to overload
* PR #5193: Fix CI failure due to missing files when installed
* PR #5213: Fix ``.strip()`` to strip all whitespace characters
* PR #5216: Fix namedtuple mistreated by dispatcher as simple tuple
* PR #5222: Fix compiler warnings due to missing (void)
* PR #5232: Fixes a bad import that breaks master
* PR #5239: fix test discovery for unittest
* PR #5247: Continue PR #5126
* PR #5250: Part fix/5098
* PR #5252: Trivially fix SyntaxWarning
* PR #5276: Add prange variant to has_no_side_effect.
* PR #5278: fix error messages in test_array_reductions
* PR #5310: PR #3897 continued
* PR #5313: Continues PR #4376
* PR #5318: Remove AUTHORS file reference from MANIFEST.in
* PR #5327: Add warning if FNV hashing is found as the default for CPython.
* PR #5338: Remove refcount pruning pass
* PR #5345: Disable test failing due to removed pass.
* PR #5357: Small fix to have llvm and asm highlighted properly
* PR #5361: 5081 typed.dict
* PR #5431: Add tolerance to numba extension module entrypoints.
* PR #5432: Fix code causing compiler warnings.
* PR #5445: Remove undefined variable
* PR #5454: Don't warn for numba.experimental.jitclass
* PR #5459: Fixes issue 5448
* PR #5480: Fix for #5477, literal_unroll KeyError searching for getitems
* PR #5485: Show the offending module in "no direct replacement" error message
* PR #5488: Add missing ``numba.config`` shim
* PR #5495: Fix missing null initializer for variable after phi strip
* PR #5498: Make the shim deprecation warnings work on python 3.6 too
* PR #5505: Better error message if __init__ returns value
* PR #5527: Attempt to fix #5518
* PR #5529: PR #5473 continued
* PR #5532: Make ``numba.<mod>`` available without an import
* PR #5542: Fixes RC2 module shim bug
* PR #5548: Fix #5537 Removed reference to ``platform.linux_distribution``
* PR #5555: Fix #5515 by reverting changes to ArrayAnalysis
* PR #5557: First-class function call cannot use keyword arguments
* PR #5569: Fix RewriteConstGetitems not registering calltype for new expr
* PR #5571: Pin down llvmlite requirement

CUDA Enhancements/Fixes:

* PR #5061: Prevent kernel launch with no configuration, remove autotuner
* PR #5085: Support other itemsizes with view
* PR #5099: Fix #5073: Slices of dynamic shared memory all alias
* PR #5104: Add a list of libraries that support __cuda_array_interface__
* PR #5136: CUDA: Enable asynchronous operations on the default stream
* PR #5144: Fix #4875: Make #2655 test with debug expect to pass
* PR #5189: __cuda_array_interface__ not requiring context
* PR #5253: Coerce ``dtype`` to ``numpy.dtype``
* PR #5257: boundscheck fix
* PR #5319: Make user facing error string use abs path not rel.
* PR #5323: Document lifetime semantics of CUDA Array Interface
* PR #5328: NBEP 7: External Memory Management Plugin Interface
* PR #5343: Fix cuda spoof
* PR #5400: Fix #4954, and some other small CUDA testsuite fixes
* PR #5436: Fix #5408: test_set_registers_57 fails on Maxwell
* PR #5519: CUDA: Silence the test suite - Fix #4809, remove autojit, delete
  prints

Documentation Updates:

* PR #4957: Add notes on overwriting gufunc inputs to docs
* PR #5059: Docs: Explain how to use Memcheck with Numba, fixups in CUDA
  documentation
* PR #5106: Add SciPy 2019 talks to docs
* PR #5147: Update master for 0.48.0 updates
* PR #5155: Explain what inlining at Numba IR level will do
* PR #5161: Fix README.rst formatting
* PR #5207: Remove AUTHORS list
* PR #5249: fix target path for See also
* PR #5262: fix typo in inlining docs
* PR #5270: fix 'see also' in typeddict docs
* PR #5280: Added some hints for debugging entry points.
* PR #5297: Update docs with intro to {g,}ufuncs.
* PR #5326: Update installation docs with OpenMP requirements.
* PR #5346: Docs: use sphinx_rtd_theme
* PR #5366: Remove reference to Python 2.7 in install check output
* PR #5423: Fix #5421: Add notes on printing in CUDA kernels
* PR #5438: Update package deps for doc building.
* PR #5440: Bump deprecation notices.
* PR #5443: Fix #5196: Docs: assert in CUDA only enabled for debug
* PR #5450: Docs: remove numbers and add titles to TOC
* PR #5453: fix building docs on windows
* PR #5489: docs: fix rendering of nested bulleted list

CI updates:

* PR #5314: Update the image used in Azure CI for OSX.
* PR #5360: Remove Travis CI badge.

Authors:

* Aaron Meurer
* Andreas Sodeur
* Antonio Russo
* Arik Funke
* Eric Wieser
* Ethan Pronovost
* Gabriel Majeri
* Graham Markall
* Greg Jennings
* Guilherme Leobas
* hdf
* Henry Schreiner
* Ivan Butygin
* Jacques Gaudin
* Jim Pivarski
* John Kirkham
* Leo Fang
* Lucio Fernandez-Arjona
* Mads R. B. Kristensen
* Marcin Tolysz
* Mike Williams
* Pearu Peterson
* Ravi Teja Gutta
* Rohit Sanjay
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)
* Vladimir Lukyanov


Version 0.48.0 (Jan 27, 2020)
-----------------------------

This release is particularly small as it was present to catch anything that
missed the 0.47.0 deadline (the deadline deliberately coincided with the end of
support for Python 2.7). The next release will be considerably larger.

The core changes in this release are dominated by the start of the clean up
needed for the end of Python 2.7 support, improvements to the CUDA target and
support for numerous additional unicode string methods.

Enhancements from user contributed PRs (with thanks!):

* Brian Wignall fixed more spelling typos in #4998.
* Denis Smirnov added support for string methods ``capitalize`` (#4823),
  ``casefold`` (#4824), ``swapcase`` (#4825), ``rsplit`` (#4834), ``partition``
  (#4845) and ``splitlines`` (#4849).
* Elena Totmenina extended support for string methods ``startswith`` (#4867) and
  added ``endswith`` (#4868).
* Eric Wieser made ``type_callable`` return the decorated function itself in
  #4760
* Ethan Pronovost added support for ``np.argwhere`` in #4617
* Graham Markall contributed a large number of CUDA enhancements and fixes,
  namely:

  * #5068: Remove Python 3.4 backports from utils
  * #4975: Make ``device_array_like`` create contiguous arrays (Fixes #4832)
  * #5023: Don't launch ForAll kernels with 0 elements (Fixes #5017)
  * #5016: Fix various issues in CUDA library search (Fixes #4979)
  * #5014: Enable use of records and bools for shared memory, remove ddt, add
    additional transpose tests
  * #4964: Fix #4628: Add more appropriate typing for CUDA device arrays
  * #5007: test_consuming_strides: Keep dev array alive
  * #4997: State that CUDA Toolkit 8.0 required in docs

* James Bourbeau added the Python 3.8 classifier to setup.py in #5027.
* John Kirkham added a clarification to the ``__cuda_array_interface__``
  documentation in #5049.
* Leo Fang Fixed an indexing problem in ``dummyarray`` in #5012.
* Marcel Bargull fixed a build and test issue for Python 3.8 in #5029.
* Maria Rubtsov added support for string methods ``isdecimal`` (#4842),
  ``isdigit`` (#4843), ``isnumeric`` (#4844) and ``replace`` (#4865).

General Enhancements:

* PR #4760: Make type_callable return the decorated function
* PR #5010: merge string prs

  This merge PR included the following:

  * PR #4823: Implement str.capitalize() based on CPython
  * PR #4824: Implement str.casefold() based on CPython
  * PR #4825: Implement str.swapcase() based on CPython
  * PR #4834: Implement str.rsplit() based on CPython
  * PR #4842: Implement str.isdecimal
  * PR #4843: Implement str.isdigit
  * PR #4844: Implement str.isnumeric
  * PR #4845: Implement str.partition() based on CPython
  * PR #4849: Implement str.splitlines() based on CPython
  * PR #4865: Implement str.replace
  * PR #4867: Functionality extension str.startswith() based on CPython
  * PR #4868: Add functionality for str.endswith()

* PR #5039: Disable help messages.
* PR #4617: Add coverage for ``np.argwhere``

Fixes:

* PR #4724: Only use lives (and not aliases) to create post parfor live set.
* PR #4998: Fix more spelling typos
* PR #5024: Propagate semantic constants ahead of static rewrites.
* PR #5027: Add Python 3.8 classifier to setup.py
* PR #5046: Update setup.py and buildscripts for dependency requirements
* PR #5053: Convert from arrays to names in define() and don't invalidate for
  multiple consistent defines.
* PR #5058: Permit mixed int types in wrap_index
* PR #5078: Catch the use of global typed-list in JITed functions
* PR #5092: Fix #5087, bug in bytecode analysis.

CUDA Enhancements/Fixes:

* PR #4964: Fix #4628: Add more appropriate typing for CUDA device arrays
* PR #4975: Make ``device_array_like`` create contiguous arrays (Fixes #4832)
* PR #4997: State that CUDA Toolkit 8.0 required in docs
* PR #5007: test_consuming_strides: Keep dev array alive
* PR #5012: Fix IndexError when accessing the "-1" element of dummyarray
* PR #5014: Enable use of records and bools for shared memory, remove ddt, add
  additional transpose tests
* PR #5016: Fix various issues in CUDA library search (Fixes #4979)
* PR #5023: Don't launch ForAll kernels with 0 elements (Fixes #5017)
* PR #5068: Remove Python 3.4 backports from utils

Documentation Updates:

* PR #5049: Clarify what dictionary means
* PR #5062: Update docs for updated version requirements
* PR #5090: Update deprecation notices for 0.48.0

CI updates:

* PR #5029: Install optional dependencies for Python 3.8 tests
* PR #5040: Drop Py2.7 and Py3.5 from public CI
* PR #5048: Fix CI py38

Authors:

* Brian Wignall
* Denis Smirnov
* Elena Totmenina
* Eric Wieser
* Ethan Pronovost
* Graham Markall
* James Bourbeau
* John Kirkham
* Leo Fang
* Marcel Bargull
* Maria Rubtsov
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)


Version 0.47.0  (Jan 2, 2020)
-----------------------------

This release expands the capability of Numba in a number of important areas and
is also significant as it is the last major point release with support for
Python 2 and Python 3.5 included. The next release (0.48.0) will be for Python
3.6+ only!  (This follows NumPy's deprecation schedule as specified in
`NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.)

Highlights of core feature changes include:

* Full support for Python 3.8 (Siu Kwan Lam)
* Opt-in bounds checking (Aaron Meurer)
* Support for ``map``, ``filter`` and ``reduce`` (Stuart Archibald)

Intel also kindly sponsored research and development that lead to some exciting
new features:

* Initial support for basic ``try``/``except`` use (Siu Kwan Lam)
* The ability to pass functions created from closures/lambdas as arguments
  (Stuart Archibald)
* ``sorted`` and ``list.sort()`` now accept the ``key`` argument (Stuart
  Archibald and Siu Kwan Lam)
* A new compiler pass triggered through the use of the function
  ``numba.literal_unroll`` which permits iteration over heterogeneous tuples
  and constant lists of constants. (Stuart Archibald)

Enhancements from user contributed PRs (with thanks!):

* Ankit Mahato added a reference to a new talk on Numba at PyCon India 2019 in
  #4862
* Brian Wignall kindly fixed some spelling mistakes and typos in #4909
* Denis Smirnov wrote numerous methods to considerable enhance string support
  including:

  * ``str.rindex()`` in #4861
  * ``str.isprintable()`` in #4836
  * ``str.index()`` in #4860
  * ``start/end`` parameters for ``str.find()`` in #4866
  * ``str.isspace()`` in #4835
  * ``str.isidentifier()`` #4837
  * ``str.rpartition()`` in #4841
  * ``str.lower()`` and ``str.islower()`` in #4651

* Elena Totmenina implemented both ``str.isalnum()``, ``str.isalpha()`` and
  ``str.isascii`` in #4839, #4840 and #4847 respectively.
* Eric Larson fixed a bug in literal comparison in #4710
* Ethan Pronovost updated the ``np.arange`` implementation in #4770 to allow
  the use of the ``dtype`` key word argument and also added ``bool``
  implementations for several types in #4715.
* Graham Markall fixed some issues with the CUDA target, namely:

  * #4931: Added physical limits for CC 7.0 / 7.5 to CUDA autotune
  * #4934: Fixed bugs in TestCudaWarpOperations
  * #4938: Improved errors / warnings for the CUDA vectorize decorator

* Guilherme Leobas fixed a typo in the ``urem`` implementation in #4667
* Isaac Virshup contributed a number of patches that fixed bugs, added support
  for more NumPy functions and enhanced Python feature support. These
  contributions included:

  * #4729: Allow array construction with mixed type shape tuples
  * #4904: Implementing ``np.lcm``
  * #4780: Implement np.gcd and math.gcd
  * #4779: Make slice constructor more similar to python.
  * #4707: Added support for slice.indices
  * #4578: Clarify numba ufunc supported features

* James Bourbeau fixed some issues with tooling, #4794 add ``setuptools`` as a
  dependency and #4501 add pre-commit hooks for ``flake8`` compliance.
* Leo Fang made ``numba.dummyarray.Array`` iterable in #4629
* Marc Garcia fixed the ``numba.jit`` parameter name signature_or_function in
  #4703
* Marcelo Duarte Trevisani patched the llvmlite requirement to ``>=0.30.0`` in
  #4725
* Matt Cooper fixed a long standing CI problem in #4737 by remove maxParallel
* Matti Picus fixed an issue with ``collections.abc`` in #4734
  from Azure Pipelines.
* Rob Ennis patched a bug in ``np.interp`` ``float32`` handling in #4911
* VDimir fixed a bug in array transposition layouts in #4777 and re-enabled and
  fixed some idle tests in #4776.
* Vyacheslav Smirnov Enable support for `str.istitle()`` in #4645

General Enhancements:

* PR #4432: Bounds checking
* PR #4501: Add pre-commit hooks
* PR #4536: Handle kw args in inliner when callee is a function
* PR #4599: Permits closures to become functions, enables map(), filter()
* PR #4611: Implement method title() for unicode based on Cpython
* PR #4645: Enable support for istitle() method for unicode string
* PR #4651: Implement str.lower() and str.islower()
* PR #4652: Implement str.rfind()
* PR #4695: Refactor `overload*` and support `jit_options` and `inline`
* PR #4707: Added support for slice.indices
* PR #4715: Add `bool` overload for several types
* PR #4729: Allow array construction with mixed type shape tuples
* PR #4755: Python3.8 support
* PR #4756: Add parfor support for ndarray.fill.
* PR #4768: Update typeconv error message to ask for sys.executable.
* PR #4770: Update `np.arange` implementation with `@overload`
* PR #4779: Make slice constructor more similar to python.
* PR #4780: Implement np.gcd and math.gcd
* PR #4794: Add setuptools as a dependency
* PR #4802: put git hash into build string
* PR #4803: Better compiler error messages for improperly used reduction
  variables.
* PR #4817: Typed list implement and expose allocation
* PR #4818: Typed list faster copy
* PR #4835: Implement str.isspace() based on CPython
* PR #4836: Implement str.isprintable() based on CPython
* PR #4837: Implement str.isidentifier() based on CPython
* PR #4839: Implement str.isalnum() based on CPython
* PR #4840: Implement str.isalpha() based on CPython
* PR #4841: Implement str.rpartition() based on CPython
* PR #4847: Implement str.isascii() based on CPython
* PR #4851: Add graphviz output for FunctionIR
* PR #4854: Python3.8 looplifting
* PR #4858: Implement str.expandtabs() based on CPython
* PR #4860: Implement str.index() based on CPython
* PR #4861: Implement str.rindex() based on CPython
* PR #4866: Support params start/end for str.find()
* PR #4874: Bump to llvmlite 0.31
* PR #4896: Specialise arange dtype on arch + python version.
* PR #4902: basic support for try except
* PR #4904: Implement np.lcm
* PR #4910: loop canonicalisation and type aware tuple unroller/loop body
  versioning passes
* PR #4961: Update hash(tuple) for Python 3.8.
* PR #4977: Implement sort/sorted with key.
* PR #4987: Add `is_internal` property to all Type classes.

Fixes:

* PR #4090: Update to LLVM8 memset/memcpy intrinsic
* PR #4582: Convert sub to add and div to mul when doing the reduction across
  the per-thread reduction array.
* PR #4648: Handle 0 correctly as slice parameter.
* PR #4660: Remove multiply defined variables from all blocks' equivalence sets.
* PR #4672: Fix pickling of dufunc
* PR #4710: BUG: Comparison for literal
* PR #4718: Change get_call_table to support intermediate Vars.
* PR #4725: Requires  llvmlite >=0.30.0
* PR #4734: prefer to import from collections.abc
* PR #4736: fix flake8 errors
* PR #4776: Fix and enable idle tests from test_array_manipulation
* PR #4777: Fix transpose output array layout
* PR #4782: Fix issue with SVML (and knock-on function resolution effects).
* PR #4785: Treat 0d arrays like scalars.
* PR #4787: fix missing incref on flags
* PR #4789: fix typos in numba/targets/base.py
* PR #4791: fix typos
* PR #4811: fix spelling in now-failing tests
* PR #4852: windowing test should check equality only up to double precision
  errors
* PR #4881: fix refining list by using extend on an iterator
* PR #4882: Fix return type in arange and zero step size handling.
* PR #4885: suppress spurious RuntimeWarning about ufunc sizes
* PR #4891: skip the xfail test for now.  Py3.8 CFG refactor seems to have
  changed the test case
* PR #4892: regex needs to accept singular form of "argument"
* PR #4901: fix typed list equals
* PR #4909: Fix some spelling typos
* PR #4911: np.interp bugfix for float32 handling
* PR #4920: fix creating list with JIT disabled
* PR #4921: fix creating dict with JIT disabled
* PR #4935: Better handling of prange with multiple reductions on the same
  variable.
* PR #4946: Improve the error message for `raise <string>`.
* PR #4955: Move overload of literal_unroll to avoid circular dependency that
  breaks Python 2.7
* PR #4962: Fix test error on windows
* PR #4973: Fixes a bug in the relabelling logic in literal_unroll.
* PR #4978: Fix overload_method problem with stararg
* PR #4981: Add ind_to_const to enable fewer equivalence classes.
* PR #4991: Continuation of #4588 (Let dead code removal handle removing more of
  the unneeded code after prange conversion to parfor)
* PR #4994: Remove xfail for test which has since had underlying issue fixed.
* PR #5018: Fix #5011.
* PR #5019: skip pycc test on Python 3.8 + macOS because of distutils issue

CUDA Enhancements/Fixes:

* PR #4629: Make numba.dummyarray.Array iterable
* PR #4675: Bump cuda array interface to version 2
* PR #4741: Update choosing the "CUDA_PATH" for windows
* PR #4838: Permit ravel('A') for contig device arrays in CUDA target
* PR #4931: Add physical limits for CC 7.0 / 7.5 to autotune
* PR #4934: Fix fails in TestCudaWarpOperations
* PR #4938: Improve errors / warnings for cuda vectorize decorator

Documentation Updates:

* PR #4418: Directed graph task roadmap
* PR #4578: Clarify numba ufunc supported features
* PR #4655: fix sphinx build warning
* PR #4667: Fix typo on urem implementation
* PR #4669: Add link to ParallelAccelerator paper.
* PR #4703: Fix numba.jit parameter name signature_or_function
* PR #4862: Addition of PyCon India 2019 talk on Numba
* PR #4947: Document jitclass with numba.typed use.
* PR #4958: Add docs for `try..except`
* PR #4993: Update deprecations for 0.47

CI Updates:

* PR #4737: remove maxParallel from Azure Pipelines
* PR #4767: pin to 2.7.16 for py27 on osx
* PR #4781: WIP/runtest cf pytest

Authors:

* Aaron Meurer
* Ankit Mahato
* Brian Wignall
* Denis Smirnov
* Ehsan Totoni (core dev)
* Elena Totmenina
* Eric Larson
* Ethan Pronovost
* Giovanni Cavallin
* Graham Markall
* Guilherme Leobas
* Isaac Virshup
* James Bourbeau
* Leo Fang
* Marc Garcia
* Marcelo Duarte Trevisani
* Matt Cooper
* Matti Picus
* Rob Ennis
* Rujal Desai
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* VDimir
* Valentin Haenel (core dev)
* Vyacheslav Smirnov


Version 0.46.0
--------------

This release significantly reworked one of the main parts of Numba, the compiler
pipeline, to make it more extensible and easier to use. The purpose of this was
to continue enhancing Numba's ability for use as a compiler toolkit. In a
similar vein, Numba now has an extension registration mechanism to allow other
Numba-using projects to automatically have their Numba JIT compilable functions
discovered. There were also a number of other related compiler toolkit
enhancement added along with some more NumPy features and a lot of bug fixes.

This release has updated the CUDA Array Interface specification to version 2,
which clarifies the `strides` attribute for C-contiguous arrays and specifies
the treatment for zero-size arrays. The implementation in Numba has been
changed and may affect downstream packages relying on the old behavior
(see issue #4661).

Enhancements from user contributed PRs (with thanks!):

* Aaron Meurer fixed some Python issues in the code base in #4345 and #4341.
* Ashwin Srinath fixed a CUDA performance bug via #4576.
* Ethan Pronovost added support for triangular indices functions in #4601 (the
  NumPy functions ``tril_indices``, ``tril_indices_from``, ``triu_indices``, and
  ``triu_indices_from``).
* Gerald Dalley fixed a tear down race occurring in Python 2.
* Gregory R. Lee fixed the use of deprecated ``inspect.getargspec``.
* Guilherme Leobas contributed five PRs, adding support for ``np.append`` and
  ``np.count_nonzero`` in #4518 and #4386. The typed List was fixed to accept
  unsigned integers in #4510. #4463 made a fix to NamedTuple internals and #4397
  updated the docs for ``np.sum``.
* James Bourbeau added a new feature to permit the automatic application of the
  `jit` decorator to a whole module in #4331. Also some small fixes to the docs
  and the code base were made in #4447 and #4433, and a fix to inplace array
  operation in #4228.
* Jim Crist fixed a bug in the rendering of patched errors in #4464.
* Leo Fang updated the CUDA Array Interface contract in #4609.
* Pearu Peterson added support for Unicode based NumPy arrays in #4425.
* Peter Andreas Entschev fixed a CUDA concurrency bug in #4581.
* Lucio Fernandez-Arjona extended Numba's ``np.sum`` support to now accept the
  ``dtype`` kwarg in #4472.
* Pedro A. Morales Maries added support for ``np.cross`` in #4128 and also added
  the necessary extension ``numba.numpy_extensions.cross2d`` in #4595.
* David Hoese, Eric Firing, Joshua Adelman, and Juan Nunez-Iglesias all made
  documentation fixes in #4565, #4482, #4455, #4375 respectively.
* Vyacheslav Smirnov and Rujal Desai enabled support for ``count()`` on unicode
  strings in #4606.

General Enhancements:

* PR #4113: Add rewrite for semantic constants.
* PR #4128: Add np.cross support
* PR #4162: Make IR comparable and legalize it.
* PR #4208: R&D inlining, jitted and overloaded.
* PR #4331: Automatic JIT of called functions
* PR #4353: Inspection tool to check what numba supports
* PR #4386: Implement np.count_nonzero
* PR #4425: Unicode array support
* PR #4427: Entrypoints for numba extensions
* PR #4467: Literal dispatch
* PR #4472: Allow dtype input argument in np.sum
* PR #4513: New compiler.
* PR #4518: add support for np.append
* PR #4554: Refactor NRT C-API
* PR #4556: 0.46 scheduled deprecations
* PR #4567: Add env var to disable performance warnings.
* PR #4568: add np.array_equal support
* PR #4595: Implement numba.cross2d
* PR #4601: Add triangular indices functions
* PR #4606: Enable support for count() method for unicode string

Fixes:

* PR #4228: Fix inplace operator error for arrays
* PR #4282: Detect and raise unsupported on generator expressions
* PR #4305: Don't allow the allocation of mutable objects written into a
  container to be hoisted.
* PR #4311: Avoid deprecated use of inspect.getargspec
* PR #4328:  Replace GC macro with function call
* PR #4330: Loosen up typed container casting checks
* PR #4341: Fix some coding lines at the top of some files (utf8 -> utf-8)
* PR #4345: Replace "import \*" with explicit imports in numba/types
* PR #4346: Fix incorrect alg in isupper for ascii strings.
* PR #4349: test using jitclass in typed-list
* PR #4361: Add allocation hoisting info to LICM section at diagnostic L4
* PR #4366: Offset search box to avoid wrapping on some pages with Safari.
  Fixes #4365.
* PR #4372: Replace all "except BaseException" with "except Exception".
* PR #4407: Restore the "free" conda channel for NumPy 1.10 support.
* PR #4408: Add lowering for constant bytes.
* PR #4409: Add exception chaining for better error context
* PR #4411: Name of type should not contain user facing description for debug.
* PR #4412: Fix #4387. Limit the number of return types for recursive functions
* PR #4426: Fixed two module teardown races in py2.
* PR #4431: Fix and test numpy.random.random_sample(n) for np117
* PR #4463: NamedTuple - Raises an error on non-iterable elements
* PR #4464: Add a newline in patched errors
* PR #4474: Fix liveness for remove dead of parfors (and other IR extensions)
* PR #4510: Make List.__getitem__ accept unsigned parameters
* PR #4512: Raise specific error at typing time for iteration on >1D array.
* PR #4532: Fix static_getitem with Literal type as index
* PR #4547: Update to inliner cost model information.
* PR #4557: Use specific random number seed when generating arbitrary test data
* PR #4559: Adjust test timeouts
* PR #4564: Skip unicode array tests on ppc64le that trigger an LLVM bug
* PR #4621: Fix packaging issue due to missing numba/cext
* PR #4623: Fix issue 4520 due to storage model mismatch
* PR #4644: Updates for llvmlite 0.30.0

CUDA Enhancements/Fixes:

* PR #4410: Fix #4111. cudasim mishandling recarray
* PR #4576: Replace use of `np.prod` with `functools.reduce` for computing size
  from shape
* PR #4581: Prevent taking the GIL in ForAll
* PR #4592: Fix #4589.  Just pass NULL for b2d_func for constant dynamic
  sharedmem
* PR #4609: Update CUDA Array Interface & Enforce Numba compliance
* PR #4619: Implement math.{degrees, radians} for the CUDA target.
* PR #4675: Bump cuda array interface to version 2

Documentation Updates:

* PR #4317: Add docs for ARMv8/AArch64
* PR #4318: Add supported platforms to the docs.  Closes #4316
* PR #4375: Add docstrings to inspect methods
* PR #4388: Update Python 2.7 EOL statement
* PR #4397: Add note about np.sum
* PR #4447: Minor parallel performance tips edits
* PR #4455: Clarify docs for typed dict with regard to arrays
* PR #4482: Fix example in guvectorize docstring.
* PR #4541: fix two typos in architecture.rst
* PR #4548: Document numba.extending.intrinsic and inlining.
* PR #4565: Fix typo in jit-compilation docs
* PR #4607: add dependency list to docs
* PR #4614: Add documentation for implementing new compiler passes.

CI Updates:

* PR #4415: Make 32bit incremental builds on linux not use free channel
* PR #4433: Removes stale azure comment
* PR #4493: Fix Overload Inliner wrt CUDA Intrinsics
* PR #4593: Enable Azure CI batching

Contributors:

* Aaron Meurer
* Ashwin Srinath
* David Hoese
* Ehsan Totoni (core dev)
* Eric Firing
* Ethan Pronovost
* Gerald Dalley
* Gregory R. Lee
* Guilherme Leobas
* James Bourbeau
* Jim Crist
* Joshua Adelman
* Juan Nunez-Iglesias
* Leo Fang
* Lucio Fernandez-Arjona
* Pearu Peterson
* Pedro A. Morales Marie
* Peter Andreas Entschev
* Rujal Desai
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)
* Vyacheslav Smirnov


Version 0.45.1
--------------

This patch release addresses some regressions reported in the 0.45.0 release and
adds support for NumPy 1.17:

* PR #4325: accept scalar/0d-arrays
* PR #4338: Fix #4299. Parfors reduction vars not deleted.
* PR #4350: Use process level locks for fork() only.
* PR #4354: Try to fix #4352.
* PR #4357: Fix np1.17 isnan, isinf, isfinite ufuncs
* PR #4363: Fix np.interp for np1.17 nan handling
* PR #4371: Fix nump1.17 random function non-aliasing

Contributors:

* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)
* Valentin Haenel (core dev)


Version 0.45.0
--------------

In this release, Numba gained an experimental :ref:`numba.typed.List
<feature-typed-list>` container as a future replacement of the :ref:`reflected
list <feature-reflected-list>`. In addition, functions decorated with
``parallel=True`` can now be cached to reduce compilation overhead associated
with the auto-parallelization.


Enhancements from user contributed PRs (with thanks!):

* James Bourbeau added the Numba version to reportable error messages in #4227,
  added the ``signature`` parameter to ``inspect_types`` in #4200, improved the
  docstring of ``normalize_signature`` in #4205, and fixed #3658 by adding
  reference counting to ``register_dispatcher`` in #4254

* Guilherme Leobas implemented the dominator tree and dominance frontier
  algorithms in #4216 and #4149, respectively.

* Nick White fixed the issue with ``round`` in the CUDA target in #4137.

* Joshua Adelman added support for determining if a value is in a `range`
  (i.e.  ``x in range(...)``) in #4129, and added windowing functions
  (``np.bartlett``, ``np.hamming``, ``np.blackman``, ``np.hanning``,
  ``np.kaiser``) from NumPy in #4076.

* Lucio Fernandez-Arjona added support for ``np.select`` in #4077

* Rob Ennis added support for ``np.flatnonzero`` in #4157

* Keith Kraus extended the ``__cuda_array_interface__`` with an optional mask
  attribute in #4199.

* Gregory R. Lee replaced deprecated use of ``inspect.getargspec`` in #4311.


General Enhancements:

* PR #4328: Replace GC macro with function call
* PR #4311: Avoid deprecated use of inspect.getargspec
* PR #4296: Slacken window function testing tol on ppc64le
* PR #4254: Add reference counting to register_dispatcher
* PR #4239: Support len() of multi-dim arrays in array analysis
* PR #4234: Raise informative error for np.kron array order
* PR #4232: Add unicodetype db, low level str functions and examples.
* PR #4229: Make hashing cacheable
* PR #4227: Include numba version in reportable error message
* PR #4216: Add dominator tree
* PR #4200: Add signature parameter to inspect_types
* PR #4196: Catch missing imports of internal functions.
* PR #4180: Update use of unlowerable global message.
* PR #4166: Add tests for PR #4149
* PR #4157: Support for np.flatnonzero
* PR #4149: Implement dominance frontier for SSA for the Numba IR
* PR #4148: Call branch pruning in inline_closure_call()
* PR #4132: Reduce usage of inttoptr
* PR #4129: Support contains for range
* PR #4112: better error messages for np.transpose and tuples
* PR #4110: Add range attrs, start, stop, step
* PR #4077: Add np select
* PR #4076: Add numpy windowing functions support (np.bartlett, np.hamming,
  np.blackman, np.hanning, np.kaiser)
* PR #4095: Support ir.Global/FreeVar in find_const()
* PR #3691: Make TypingError abort compiling earlier
* PR #3646: Log internal errors encountered in typeinfer

Fixes:

* PR #4303: Work around scipy bug 10206
* PR #4302: Fix flake8 issue on master
* PR #4301: Fix integer literal bug in np.select impl
* PR #4291: Fix pickling of jitclass type
* PR #4262: Resolves #4251 - Fix bug in reshape analysis.
* PR #4233: Fixes issue revealed by #4215
* PR #4224: Fix #4223. Looplifting error due to StaticSetItem in objectmode
* PR #4222: Fix bad python path.
* PR #4178: Fix unary operator overload, check with unicode impl
* PR #4173: Fix return type in np.bincount with weights
* PR #4153: Fix slice shape assignment in array analysis
* PR #4152: fix status check in dict lookup
* PR #4145: Use callable instead of checking __module__
* PR #4118: Fix inline assembly support on CPU.
* PR #4088: Resolves #4075 - parfors array_analysis bug.
* PR #4085: Resolves #3314 - parfors array_analysis bug with reshape.

CUDA Enhancements/Fixes:

* PR #4199: Extend `__cuda_array_interface__` with optional mask attribute,
  bump version to 1
* PR #4137: CUDA - Fix round Builtin
* PR #4114: Support 3rd party activated CUDA context

Documentation Updates:

* PR #4317: Add docs for ARMv8/AArch64
* PR #4318: Add supported platforms to the docs. Closes #4316
* PR #4295: Alter deprecation schedules
* PR #4253: fix typo in pysupported docs
* PR #4252: fix typo on repomap
* PR #4241: remove unused import
* PR #4240: fix typo in jitclass docs
* PR #4205: Update return value order in normalize_signature docstring
* PR #4237: Update doc links to point to latest not dev docs.
* PR #4197: hyperlink repomap
* PR #4170: Clarify docs on accumulating into arrays in prange
* PR #4147: fix docstring for DictType iterables
* PR #3951: A guide to overloading

CI Updates:

* PR #4300: AArch64 has no faulthandler package
* PR #4273: pin to MKL BLAS for testing to get consistent results
* PR #4209: Revert previous network tol patch and try with conda config
* PR #4138: Remove tbb before Azure test only on Python 3, since it was already
  removed for Python 2

Contributors:

* Ehsan Totoni (core dev)
* Gregory R. Lee
* Guilherme Leobas
* James Bourbeau
* Joshua L. Adelman
* Keith Kraus
* Lucio Fernandez-Arjona
* Nick White
* Rob Ennis
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)


Version 0.44.1
--------------

This patch release addresses some regressions reported in the 0.44.0 release:

- PR #4165: Fix #4164 issue with NUMBAPRO_NVVM.
- PR #4172: Abandon branch pruning if an arg name is redefined. (Fixes #4163)
- PR #4183: Fix #4156. Problem with defining in-loop variables.


Version 0.44.0
--------------

IMPORTANT: In this release a few significant deprecations (and some less
significant ones) are being made, users are encouraged to read the related
documentation.

General enhancements in this release include:

- Numba is backed by LLVM 8 on all platforms apart from ppc64le, which, due to
  bugs, remains on the LLVM 7.x series.
- Numba's dictionary support now includes type inference for keys and values.
- The .view() method now works for NumPy scalar types.
- Newly supported NumPy functions added: np.delete, np.nanquantile, np.quantile,
  np.repeat, np.shape.

In addition considerable effort has been made to fix some long standing bugs and
a large number of other bugs, the "Fixes" section is very large this time!

Enhancements from user contributed PRs (with thanks!):

- Max Bolingbroke added support for the selective use of ``fastmath`` flags in
  #3847.
- Rob Ennis made min() and max() work on iterables in #3820 and added
  np.quantile and np.nanquantile in #3899.
- Sergey Shalnov added numerous unicode string related features, zfill in #3978,
  ljust in #4001, rjust and center in #4044 and strip, lstrip and rstrip in
  #4048.
- Guilherme Leobas added support for np.delete in #3890
- Christoph Deil exposed the Numba CLI via ``python -m numba`` in #4066 and made
  numerous documentation fixes.
- Leo Schwarz wrote the bulk of the code for jitclass default constructor
  arguments in #3852.
- Nick White enhanced the CUDA backend to use min/max PTX instructions where
  possible in #4054.
- Lucio Fernandez-Arjona implemented the unicode string ``__mul__`` function in
  #3952.
- Dimitri Vorona wrote the bulk of the code to implement getitem and setitem for
  jitclass in #3861.

General Enhancements:

* PR #3820: Min max on iterables
* PR #3842: Unicode type iteration
* PR #3847: Allow fine-grained control of fastmath flags to partially address #2923
* PR #3852: Continuation of PR #2894
* PR #3861: Continuation of PR #3730
* PR #3890: Add support for np.delete
* PR #3899: Support for np.quantile and np.nanquantile
* PR #3900: Fix 3457 :: Implements np.repeat
* PR #3928: Add .view() method for NumPy scalars
* PR #3939: Update icc_rt clone recipe.
* PR #3952: __mul__ for strings, initial implementation and tests
* PR #3956: Type-inferred dictionary
* PR #3959: Create a view for string slicing to avoid extra allocations
* PR #3978: zfill operation implementation
* PR #4001: ljust operation implementation
* PR #4010: Support `dict()` and `{}`
* PR #4022: Support for llvm 8
* PR #4034: Make type.Optional str more representative
* PR #4041: Deprecation warnings
* PR #4044: rjust and center operations implementation
* PR #4048: strip, lstrip and rstrip operations implementation
* PR #4066: Expose numba CLI via python -m numba
* PR #4081: Impl `np.shape` and support function for `asarray`.
* PR #4091: Deprecate the use of iternext_impl without RefType

CUDA Enhancements/Fixes:

* PR #3933: Adds `.nbytes` property to CUDA device array objects.
* PR #4011: Add .inspect_ptx() to cuda device function
* PR #4054: CUDA: Use min/max PTX Instructions
* PR #4096: Update env-vars for CUDA libraries lookup

Documentation Updates:

* PR #3867: Code repository map
* PR #3918: adding Joris' Fosdem 2019 presentation
* PR #3926: order talks on applications of Numba by date
* PR #3943: fix two small typos in vectorize docs
* PR #3944: Fixup jitclass docs
* PR #3990: mention preprint repo in FAQ. Fixes #3981
* PR #4012: Correct runtests command in contributing.rst
* PR #4043: fix typo
* PR #4047: Ambiguous Documentation fix for guvectorize.
* PR #4060: Remove remaining mentions of autojit in docs
* PR #4063: Fix annotate example in docstring
* PR #4065: Add FAQ entry explaining Numba project name
* PR #4079: Add Documentation for atomicity of typed.Dict
* PR #4105: Remove info about CUDA ENVVAR potential replacement

Fixes:

* PR #3719: Resolves issue #3528.  Adds support for slices when not using parallel=True.
* PR #3727: Remove dels for known dead vars.
* PR #3845: Fix mutable flag transmission in .astype
* PR #3853: Fix some minor issues in the C source.
* PR #3862: Correct boolean reinterpretation of data
* PR #3863: Comments out the appveyor badge
* PR #3869: fixes flake8 after merge
* PR #3871: Add assert to ir.py to help enforce correct structuring
* PR #3881: fix preparfor dtype transform for datetime64
* PR #3884: Prevent mutation of objmode fallback IR.
* PR #3885: Updates for llvmlite 0.29
* PR #3886: Use `safe_load` from pyyaml.
* PR #3887: Add tolerance to network errors by permitting conda to retry
* PR #3893: Fix casting in namedtuple ctor.
* PR #3894: Fix array inliner for multiple array definition.
* PR #3905: Cherrypick #3903 to main
* PR #3920: Raise better error if unsupported jump opcode found.
* PR #3927: Apply flake8 to the numpy related files
* PR #3935: Silence DeprecationWarning
* PR #3938: Better error message for unknown opcode
* PR #3941: Fix typing of ufuncs in parfor conversion
* PR #3946: Return variable renaming dict from inline_closurecall
* PR #3962: Fix bug in alignment computation of `Record.make_c_struct`
* PR #3967: Fix error with pickling unicode
* PR #3964: Unicode split algo versioning
* PR #3975: Add handler for unknown locale to numba -s
* PR #3991: Permit Optionals in ufunc machinery
* PR #3995: Remove assert in type inference causing poor error message.
* PR #3996: add is_ascii flag to UnicodeType
* PR #4009: Prevent zero division error in np.linalg.cond
* PR #4014: Resolves #4007.
* PR #4021: Add a more specific error message for invalid write to a global.
* PR #4023: Fix handling of titles in record dtype
* PR #4024: Do a check if a call is const before saying that an object is multiply defined.
* PR #4027: Fix issue #4020.  Turn off no_cpython_wrapper flag when compiling for…
* PR #4033: [WIP] Fixing wrong dtype of array inside reflected list #4028
* PR #4061: Change IPython cache dir name to numba_cache
* PR #4067: Delete examples/notebooks/LinearRegr.py
* PR #4070: Catch writes to global typed.Dict and raise.
* PR #4078: Check tuple length
* PR #4084: Fix missing incref on optional return None
* PR #4089: Make the warnings fixer flush work for warning comparing on type.
* PR #4094: Fix function definition finding logic for commented def
* PR #4100: Fix alignment check on 32-bit.
* PR #4104: Use PEP 508 compliant env markers for install deps

Contributors:

* Benjamin Zaitlen
* Christoph Deil
* David Hirschfeld
* Dimitri Vorona
* Ehsan Totoni (core dev)
* Guilherme Leobas
* Leo Schwarz
* Lucio Fernandez-Arjona
* Max Bolingbroke
* NanduTej
* Nick White
* Ravi Teja Gutta
* Rob Ennis
* Sergey Shalnov
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)
* Valentin Haenel (core dev)


Version 0.43.1
--------------

This is a bugfix release that provides minor changes to fix: a bug in branch
pruning, bugs in `np.interp` functionality, and also fully accommodate the
NumPy 1.16 release series.

* PR #3826: NumPy 1.16 support
* PR #3850: Refactor np.interp
* PR #3883: Rewrite pruned conditionals as their evaluated constants.

Contributors:

* Rob Ennis
* Siu Kwan Lam (core dev)
* Stuart Archibald (core dev)


Version 0.43.0
--------------

In this release, the major new features are:

- Initial support for statically typed dictionaries
- Improvements to `hash()` to match Python 3 behavior
- Support for the heapq module
- Ability to pass C structs to Numba
- More NumPy functions: asarray, trapz, roll, ptp, extract


NOTE:

The vast majority of NumPy 1.16 behaviour is supported, however
``datetime`` and ``timedelta`` use involving ``NaT`` matches the behaviour
present in earlier release. The ufunc suite has not been extending to
accommodate the two new time computation related additions present in NumPy
1.16. In addition the functions ``ediff1d`` and ``interp`` have known minor
issues in replicating outputs exactly when ``NaN``'s occur in certain input
patterns.

General Enhancements:

* PR #3563: Support for np.roll
* PR #3572: Support for np.ptp
* PR #3592: Add dead branch prune before type inference.
* PR #3598: Implement np.asarray()
* PR #3604: Support for np.interp
* PR #3607: Some simplication to lowering
* PR #3612: Exact match flag in dispatcher
* PR #3627: Support for np.trapz
* PR #3630: np.where with broadcasting
* PR #3633: Support for np.extract
* PR #3657: np.max, np.min, np.nanmax, np.nanmin - support for complex dtypes
* PR #3661: Access C Struct as Numpy Structured Array
* PR #3678: Support for str.split and str.join
* PR #3684: Support C array in C struct
* PR #3696: Add intrinsic to help debug refcount
* PR #3703: Implementations of type hashing.
* PR #3715: Port CPython3.7 dictionary for numba internal use
* PR #3716: Support inplace concat of strings
* PR #3718: Add location to ConstantInferenceError exceptions.
* PR #3720: improve error msg about invalid signature
* PR #3731: Support for heapq
* PR #3754: Updates for llvmlite 0.28
* PR #3760: Overloadable operator.setitem
* PR #3775: Support overloading operator.delitem
* PR #3777: Implement compiler support for dictionary
* PR #3791: Implement interpreter-side interface for numba dict
* PR #3799: Support refcount'ed types in numba dict

CUDA Enhancements/Fixes:

* PR #3713: Fix the NvvmSupportError message when CC too low
* PR #3722: Fix #3705: slicing error with negative strides
* PR #3755: Make cuda.to_device accept readonly host array
* PR #3773: Adapt library search to accommodate multiple locations

Documentation Updates:

* PR #3651: fix link to berryconda in docs
* PR #3668: Add Azure Pipelines build badge
* PR #3749: DOC: Clarify when prange is different from range
* PR #3771: fix a few typos
* PR #3785: Clarify use of range as function only.
* PR #3829: Add docs for typed-dict

Fixes:

* PR #3614: Resolve #3586
* PR #3618: Skip gdb tests on ARM.
* PR #3643: Remove support_literals usage
* PR #3645: Enforce and fix that AbstractTemplate.generic must be returning a Signature
* PR #3648: Fail on @overload signature mismatch.
* PR #3660: Added Ignore message to test numba.tests.test_lists.TestLists.test_mul_error
* PR #3662: Replace six with numba.six
* PR #3663: Removes coverage computation from travisci builds
* PR #3672: Avoid leaking memory when iterating over uniform tuple
* PR #3676: Fixes constant string lowering inside tuples
* PR #3677: Ensure all referenced compiled functions are linked properly
* PR #3692: Fix test failure due to overly strict test on floating point values.
* PR #3693: Intercept failed import to help users.
* PR #3694: Fix memory leak in enumerate iterator
* PR #3695: Convert return of None from intrinsic implementation to dummy value
* PR #3697: Fix for issue #3687
* PR #3701: Fix array.T analysis (fixes #3700)
* PR #3704: Fixes for overload_method
* PR #3706: Don't push call vars recursively into nested parfors. Resolves #3686.
* PR #3710: Set as non-hoistable if a mutable variable is passed to a function in a loop. Resolves #3699.
* PR #3712: parallel=True to use better builtin mechanism to resolve call types. Resolves issue #3671
* PR #3725: Fix invalid removal of dead empty list
* PR #3740: add uintp as a valid type to the tuple operator.getitem
* PR #3758: Fix target definition update in inlining
* PR #3782: Raise typing error on yield optional.
* PR #3792: Fix non-module object used as the module of a function.
* PR #3800: Bugfix for np.interp
* PR #3808: Bump macro to include VS2014 to fix py3.5 build
* PR #3809: Add debug guard to debug only C function.
* PR #3816: Fix array.sum(axis) 1d input return type.
* PR #3821: Replace PySys_WriteStdout with PySys_FormatStdout to ensure no truncation.
* PR #3830: Getitem should not return optional type
* PR #3832: Handle single string as path in find_file()

Contributors:

* Ehsan Totoni
* Gryllos Prokopis
* Jonathan J. Helmus
* Kayla Ngan
* lalitparate
* luk-f-a
* Matyt
* Max Bolingbroke
* Michael Seifert
* Rob Ennis
* Siu Kwan Lam
* Stan Seibert
* Stuart Archibald
* Todd A. Anderson
* Tao He
* Valentin Haenel


Version 0.42.1
--------------

Bugfix release to fix the incorrect hash in OSX wheel packages.
No change in source code.


Version 0.42.0
--------------

In this release the major features are:

- The capability to launch and attach the GDB debugger from within a jitted
  function.
- The upgrading of LLVM to version 7.0.0.

We added a draft of the project roadmap to the developer manual. The roadmap is
for informational purposes only as priorities and resources may change.

Here are some enhancements from contributed PRs:

- #3532. Daniel Wennberg improved the ``cuda.{pinned, mapped}`` API so that
  the associated memory is released immediately at the exit of the context
  manager.
- #3531. Dimitri Vorona enabled the inlining of jitclass methods.
- #3516. Simon Perkins added the support for passing numpy dtypes (i.e.
  ``np.dtype("int32")``) and their type constructor (i.e. ``np.int32``) into
  a jitted function.
- #3509. Rob Ennis added support for ``np.corrcoef``.

A regression issue (#3554, #3461) relating to making an empty slice in parallel
mode is resolved by #3558.

General Enhancements:

* PR #3392: Launch and attach gdb directly from Numba.
* PR #3437: Changes to accommodate LLVM 7.0.x
* PR #3509: Support for np.corrcoef
* PR #3516: Typeof dtype values
* PR #3520: Fix @stencil ignoring cval if out kwarg supplied.
* PR #3531: Fix jitclass method inlining and avoid unnecessary increfs
* PR #3538: Avoid future C-level assertion error due to invalid visibility
* PR #3543: Avoid implementation error being hidden by the try-except
* PR #3544: Add `long_running` test flag and feature to exclude tests.
* PR #3549: ParallelAccelerator caching improvements
* PR #3558: Fixes array analysis for inplace binary operators.
* PR #3566: Skip alignment tests on armv7l.
* PR #3567: Fix unifying literal types in namedtuple
* PR #3576: Add special copy routine for NumPy out arrays
* PR #3577: Fix example and docs typos for `objmode` context manager.
  reorder statements.
* PR #3580: Use alias information when determining whether it is safe to
* PR #3583: Use `ir.unknown_loc` for unknown `Loc`, as #3390 with tests
* PR #3587: Fix llvm.memset usage changes in llvm7
* PR #3596: Fix Array Analysis for Global Namedtuples
* PR #3597: Warn users if threading backend init unsafe.
* PR #3605: Add guard for writing to read only arrays from ufunc calls
* PR #3606: Improve the accuracy of error message wording for undefined type.
* PR #3611: gdb test guard needs to ack ptrace permissions
* PR #3616: Skip gdb tests on ARM.

CUDA Enhancements:

* PR #3532: Unregister temporarily pinned host arrays at once
* PR #3552: Handle broadcast arrays correctly in host->device transfer.
* PR #3578: Align cuda and cuda simulator kwarg names.

Documentation Updates:

* PR #3545: Fix @njit description in 5 min guide
* PR #3570: Minor documentation fixes for numba.cuda
* PR #3581: Fixing minor typo in `reference/types.rst`
* PR #3594: Changing `@stencil` docs to correctly reflect `func_or_mode` param
* PR #3617: Draft roadmap as of Dec 2018

Contributors:

* Aaron Critchley
* Daniel Wennberg
* Dimitri Vorona
* Dominik Stańczak
* Ehsan Totoni (core dev)
* Iskander Sharipov
* Rob Ennis
* Simon Muller
* Simon Perkins
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)


Version 0.41.0
--------------

This release adds the following major features:

* Diagnostics showing the optimizations done by ParallelAccelerator
* Support for profiling Numba-compiled functions in Intel VTune
* Additional NumPy functions: partition, nancumsum, nancumprod, ediff1d, cov,
  conj, conjugate, tri, tril, triu
* Initial support for Python 3 Unicode strings

General Enhancements:

* PR #1968: armv7 support
* PR #2983: invert mapping b/w binop operators and the operator module #2297
* PR #3160: First attempt at parallel diagnostics
* PR #3307: Adding NUMBA_ENABLE_PROFILING envvar, enabling jit event
* PR #3320: Support for np.partition
* PR #3324: Support for np.nancumsum and np.nancumprod
* PR #3325: Add location information to exceptions.
* PR #3337: Support for np.ediff1d
* PR #3345: Support for np.cov
* PR #3348: Support user pipeline class in with lifting
* PR #3363: string support
* PR #3373: Improve error message for empty imprecise lists.
* PR #3375: Enable overload(operator.getitem)
* PR #3402: Support negative indexing in tuple.
* PR #3414: Refactor Const type
* PR #3416: Optimized usage of alloca out of the loop
* PR #3424: Updates for llvmlite 0.26
* PR #3462: Add support for `np.conj/np.conjugate`.
* PR #3480: np.tri, np.tril, np.triu - default optional args
* PR #3481: Permit dtype argument as sole kwarg in np.eye

CUDA Enhancements:

* PR #3399: Add max_registers Option to cuda.jit

Continuous Integration / Testing:

* PR #3303: CI with Azure Pipelines
* PR #3309: Workaround race condition with apt
* PR #3371: Fix issues with Azure Pipelines
* PR #3362: Fix #3360: `RuntimeWarning: 'numba.runtests' found in sys.modules`
* PR #3374: Disable openmp in wheel building
* PR #3404: Azure Pipelines templates
* PR #3419: Fix cuda tests and error reporting in test discovery
* PR #3491: Prevent faulthandler installation on armv7l
* PR #3493: Fix CUDA test that used negative indexing behaviour that's fixed.
* PR #3495: Start Flake8 checking of Numba source

Fixes:

* PR #2950: Fix dispatcher to only consider contiguous-ness.
* PR #3124: Fix 3119, raise for 0d arrays in reductions
* PR #3228: Reduce redundant module linking
* PR #3329: Fix AOT on windows.
* PR #3335: Fix memory management of __cuda_array_interface__ views.
* PR #3340: Fix typo in error name.
* PR #3365: Fix the default unboxing logic
* PR #3367: Allow non-global reference to objmode() context-manager
* PR #3381: Fix global reference in objmode for dynamically created function
* PR #3382: CUDA_ERROR_MISALIGNED_ADDRESS Using Multiple Const Arrays
* PR #3384: Correctly handle very old versions of colorama
* PR #3394: Add 32bit package guard for non-32bit installs
* PR #3397: Fix with-objmode warning
* PR #3403 Fix label offset in call inline after parfor pass
* PR #3429: Fixes raising of user defined exceptions for exec(<string>).
* PR #3432: Fix error due to function naming in CI in py2.7
* PR #3444: Fixed TBB's single thread execution and test added for #3440
* PR #3449: Allow matching non-array objects in find_callname()
* PR #3455: Change getiter and iternext to not be pure. Resolves #3425
* PR #3467: Make ir.UndefinedType singleton class.
* PR #3478: Fix np.random.shuffle sideeffect
* PR #3487: Raise unsupported for kwargs given to `print()`
* PR #3488: Remove dead script.
* PR #3498: Fix stencil support for boolean as return type
* PR #3511: Fix handling make_function literals (regression of #3414)
* PR #3514: Add missing unicode != unicode
* PR #3527: Fix complex math sqrt implementation for large -ve values
* PR #3530: This adds arg an check for the pattern supplied to Parfors.
* PR #3536: Sets list dtor linkage to `linkonce_odr` to fix visibility in AOT

Documentation Updates:

* PR #3316: Update 0.40 changelog with additional PRs
* PR #3318: Tweak spacing to avoid search box wrapping onto second line
* PR #3321: Add note about memory leaks with exceptions to docs. Fixes #3263
* PR #3322: Add FAQ on CUDA + fork issue. Fixes #3315.
* PR #3343: Update docs for argsort, kind kwarg partially supported.
* PR #3357: Added mention of njit in 5minguide.rst
* PR #3434: Fix parallel reduction example in docs.
* PR #3452: Fix broken link and mark up problem.
* PR #3484: Size Numba logo in docs in em units. Fixes #3313
* PR #3502: just two typos
* PR #3506: Document string support
* PR #3513: Documentation for parallel diagnostics.
* PR #3526: Fix 5 min guide with respect to @njit decl

Contributors:

* Alex Ford
* Andreas Sodeur
* Anton Malakhov
* Daniel Stender
* Ehsan Totoni (core dev)
* Henry Schreiner
* Marcel Bargull
* Matt Cooper
* Nick White
* Nicolas Hug
* rjenc29
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Todd A. Anderson (core dev)


Version 0.40.1
--------------

This is a PyPI-only patch release to ensure that PyPI wheels can enable the
TBB threading backend, and to disable the OpenMP backend in the wheels.
Limitations of manylinux1 and variation in user environments can cause
segfaults when OpenMP is enabled on wheel builds.  Note that this release has
no functional changes for users who obtained Numba 0.40.0 via conda.

Patches:

* PR #3338: Accidentally left Anton off contributor list for 0.40.0
* PR #3374: Disable OpenMP in wheel building
* PR #3376: Update 0.40.1 changelog and docs on OpenMP backend

Version 0.40.0
--------------

This release adds a number of major features:

* A new GPU backend: kernels for AMD GPUs can now be compiled using the ROCm
  driver on Linux.
* The thread pool implementation used by Numba for automatic multithreading
  is configurable to use TBB, OpenMP, or the old "workqueue" implementation.
  (TBB is likely to become the preferred default in a future release.)
* New documentation on thread and fork-safety with Numba, along with overall
  improvements in thread-safety.
* Experimental support for executing a block of code inside a nopython mode
  function in object mode.
* Parallel loops now allow arrays as reduction variables
* CUDA improvements: FMA, faster float64 atomics on supporting hardware,
  records in const memory, and improved datatime dtype support
* More NumPy functions: vander, tri, triu, tril, fill_diagonal


General Enhancements:

* PR #3017: Add facility to support with-contexts
* PR #3033: Add support for multidimensional CFFI arrays
* PR #3122: Add inliner to object mode pipeline
* PR #3127: Support for reductions on arrays.
* PR #3145: Support for np.fill_diagonal
* PR #3151: Keep a queue of references to last N deserialized functions.  Fixes #3026
* PR #3154: Support use of list() if typeable.
* PR #3166: Objmode with-block
* PR #3179: Updates for llvmlite 0.25
* PR #3181: Support function extension in alias analysis
* PR #3189: Support literal constants in typing of object methods
* PR #3190: Support passing closures as literal values in typing
* PR #3199: Support inferring stencil index as constant in simple unary expressions
* PR #3202: Threading layer backend refactor/rewrite/reinvention!
* PR #3209: Support for np.tri, np.tril and np.triu
* PR #3211: Handle unpacking in building tuple (BUILD_TUPLE_UNPACK opcode)
* PR #3212: Support for np.vander
* PR #3227: Add NumPy 1.15 support
* PR #3272: Add MemInfo_data to runtime._nrt_python.c_helpers
* PR #3273: Refactor. Removing thread-local-storage based context nesting.
* PR #3278: compiler threadsafety lockdown
* PR #3291: Add CPU count and CFS restrictions info to numba -s.

CUDA Enhancements:

* PR #3152: Use cuda driver api to get best blocksize for best occupancy
* PR #3165: Add FMA intrinsic support
* PR #3172: Use float64 add Atomics, Where Available
* PR #3186: Support Records in CUDA Const Memory
* PR #3191: CUDA: fix log size
* PR #3198: Fix GPU datetime timedelta types usage
* PR #3221: Support datetime/timedelta scalar argument to a CUDA kernel.
* PR #3259: Add DeviceNDArray.view method to reinterpret data as a different type.
* PR #3310: Fix IPC handling of sliced cuda array.

ROCm Enhancements:

* PR #3023: Support for AMDGCN/ROCm.
* PR #3108: Add ROC info to `numba -s` output.
* PR #3176: Move ROC vectorize init to npyufunc
* PR #3177: Add auto_synchronize support to ROC stream
* PR #3178: Update ROC target documentation.
* PR #3294: Add compiler lock to ROC compilation path.
* PR #3280: Add wavebits property to the HSA Agent.
* PR #3281: Fix ds_permute types and add tests

Continuous Integration / Testing:

* PR #3091: Remove old recipes, switch to test config based on env var.
* PR #3094: Add higher ULP tolerance for products in complex space.
* PR #3096: Set exit on error in incremental scripts
* PR #3109: Add skip to test needing jinja2 if no jinja2.
* PR #3125: Skip cudasim only tests
* PR #3126: add slack, drop flowdock
* PR #3147: Improve error message for arg type unsupported during typing.
* PR #3128: Fix recipe/build for jetson tx2/ARM
* PR #3167: In build script activate env before installing.
* PR #3180: Add skip to broken test.
* PR #3216: Fix libcuda.so loading in some container setup
* PR #3224: Switch to new Gitter notification webhook URL and encrypt it
* PR #3235: Add 32bit Travis CI jobs
* PR #3257: This adds scipy/ipython back into windows conda test phase.

Fixes:

* PR #3038: Fix random integer generation to match results from NumPy.
* PR #3045: Fix #3027 - Numba reassigns sys.stdout
* PR #3059: Handler for known LoweringErrors.
* PR #3060: Adjust attribute error for NumPy functions.
* PR #3067: Abort simulator threads on exception in thread block.
* PR #3079: Implement +/-(types.boolean) Fix #2624
* PR #3080: Compute np.var and np.std correctly for complex types.
* PR #3088: Fix #3066 (array.dtype.type in prange)
* PR #3089: Fix invalid ParallelAccelerator hoisting issue.
* PR #3136: Fix #3135 (lowering error)
* PR #3137: Fix for issue3103 (race condition detection)
* PR #3142: Fix Issue #3139 (parfors reuse of reduction variable across prange blocks)
* PR #3148: Remove dead array equal @infer code
* PR #3153: Fix canonicalize_array_math typing for calls with kw args
* PR #3156: Fixes issue with missing pygments in testing and adds guards.
* PR #3168: Py37 bytes output fix.
* PR #3171: Fix #3146.  Fix CFUNCTYPE void* return-type handling
* PR #3193: Fix setitem/getitem resolvers
* PR #3222: Fix #3214.  Mishandling of POP_BLOCK in while True loop.
* PR #3230: Fixes liveness analysis issue in looplifting
* PR #3233: Fix return type difference for 32bit ctypes.c_void_p
* PR #3234: Fix types and layout for `np.where`.
* PR #3237: Fix DeprecationWarning about imp module
* PR #3241: Fix #3225.  Normalize 0nd array to scalar in typing of indexing code.
* PR #3256: Fix #3251: Move imports of ABCs to collections.abc for Python >= 3.3
* PR #3292: Fix issue3279.
* PR #3302: Fix error due to mismatching dtype

Documentation Updates:

* PR #3104: Workaround for #3098 (test_optional_unpack Heisenbug)
* PR #3132: Adds an ~5 minute guide to Numba.
* PR #3194: Fix docs RE: np.random generator fork/thread safety
* PR #3242: Page with Numba talks and tutorial links
* PR #3258: Allow users to choose the type of issue they are reporting.
* PR #3260: Fixed broken link
* PR #3266: Fix cuda pointer ownership problem with user/externally allocated pointer
* PR #3269: Tweak typography with CSS
* PR #3270: Update FAQ for functions passed as arguments
* PR #3274: Update installation instructions
* PR #3275: Note pyobject and voidptr are types in docs
* PR #3288: Do not need to call parallel optimizations "experimental" anymore
* PR #3318: Tweak spacing to avoid search box wrapping onto second line

Contributors:

* Anton Malakhov
* Alex Ford
* Anthony Bisulco
* Ehsan Totoni (core dev)
* Leonard Lausen
* Matthew Petroff
* Nick White
* Ray Donnelly
* rjenc29
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Stuart Reynolds
* Todd A. Anderson (core dev)


Version 0.39.0
--------------

Here are the highlights for the Numba 0.39.0 release.

* This is the first version that supports Python 3.7.
* With help from Intel, we have fixed the issues with SVML support (related
  issues #2938, #2998, #3006).
* List has gained support for containing reference-counted types like NumPy
  arrays and `list`.  Note, list still cannot hold heterogeneous types.
* We have made a significant change to the internal calling-convention,
  which should be transparent to most users, to allow for a future feature that
  will permitting jumping back into python-mode from a nopython-mode function.
  This also fixes a limitation to `print` that disabled its use from nopython
  functions that were deep in the call-stack.
* For CUDA GPU support, we added a `__cuda_array_interface__` following the
  NumPy array interface specification to allow Numba to consume externally
  defined device arrays.  We have opened a corresponding pull request to CuPy to
  test out the concept and be able to use a CuPy GPU array.
* The Numba dispatcher `inspect_types()` method now supports the kwarg `pretty`
  which if set to `True` will produce ANSI/HTML output, showing the annotated
  types, when invoked from ipython/jupyter-notebook respectively.
* The NumPy functions `ndarray.dot`, `np.percentile` and `np.nanpercentile`, and
  `np.unique` are now supported.
* Numba now supports the use of a per-project configuration file to permanently
  set behaviours typically set via `NUMBA_*` family environment variables.
* Support for the `ppc64le` architecture has been added.

Enhancements:

* PR #2793: Simplify and remove javascript from html_annotate templates.
* PR #2840: Support list of refcounted types
* PR #2902: Support for np.unique
* PR #2926: Enable fence for all architecture and add developer notes
* PR #2928: Making error about untyped list more informative.
* PR #2930: Add configuration file and color schemes.
* PR #2932: Fix encoding to 'UTF-8' in `check_output` decode.
* PR #2938: Python 3.7 compat: _Py_Finalizing becomes _Py_IsFinalizing()
* PR #2939: Comprehensive SVML unit test
* PR #2946: Add support for `ndarray.dot` method and tests.
* PR #2953: percentile and nanpercentile
* PR #2957: Add new 3.7 opcode support.
* PR #2963: Improve alias analysis to be more comprehensive
* PR #2984: Support for namedtuples in array analysis
* PR #2986: Fix environment propagation
* PR #2990: Improve function call matching for intrinsics
* PR #3002: Second pass at error rewrites (interpreter errors).
* PR #3004: Add numpy.empty to the list of pure functions.
* PR #3008: Augment SVML detection with llvmlite SVML patch detection.
* PR #3012: Make use of the common spelling of heterogeneous/homogeneous.
* PR #3032: Fix pycc ctypes test due to mismatch in calling-convention
* PR #3039: Add SVML detection to Numba environment diagnostic tool.
* PR #3041: This adds @needs_blas to tests that use BLAS
* PR #3056: Require llvmlite>=0.24.0

CUDA Enhancements:

* PR #2860: __cuda_array_interface__
* PR #2910: More CUDA intrinsics
* PR #2929: Add Flag To Prevent Unneccessary D->H Copies
* PR #3037: Add CUDA IPC support on non-peer-accessible devices

CI Enhancements:

* PR #3021: Update appveyor config.
* PR #3040: Add fault handler to all builds
* PR #3042: Add catchsegv
* PR #3077: Adds optional number of processes for `-m` in testing

Fixes:

* PR #2897: Fix line position of delete statement in numba ir
* PR #2905: Fix for #2862
* PR #3009: Fix optional type returning in recursive call
* PR #3019: workaround and unittest for issue #3016
* PR #3035: [TESTING] Attempt delayed removal of Env
* PR #3048: [WIP] Fix cuda tests failure on buildfarm
* PR #3054: Make test work on 32-bit
* PR #3062: Fix cuda.In freeing devary before the kernel launch
* PR #3073: Workaround #3072
* PR #3076: Avoid ignored exception due to missing globals at interpreter teardown

Documentation Updates:

* PR #2966: Fix syntax in env var docs.
* PR #2967: Fix typo in CUDA kernel layout example.
* PR #2970: Fix docstring copy paste error.

Contributors:

The following people contributed to this release.

* Anton Malakhov
* Ehsan Totoni  (core dev)
* Julia Tatz
* Matthias Bussonnier
* Nick White
* Ray Donnelly
* Siu Kwan Lam  (core dev)
* Stan Seibert  (core dev)
* Stuart Archibald  (core dev)
* Todd A. Anderson  (core dev)
* Rik-de-Kort
* rjenc29


Version 0.38.1
--------------

This is a critical bug fix release addressing:
https://github.com/numba/numba/issues/3006

The bug does not impact users using conda packages from Anaconda or Intel Python
Distribution (but it does impact conda-forge). It does not impact users of pip
using wheels from PyPI.

This only impacts a small number of users where:

 * The ICC runtime (specifically libsvml) is present in the user's environment.
 * The user is using an llvmlite statically linked against a version of LLVM
   that has not been patched with SVML support.
 * The platform is 64-bit.

The release fixes a code generation path that could lead to the production of
incorrect results under the above situation.

Fixes:

* PR #3007: Augment SVML detection with llvmlite SVML patch detection.

Contributors:

The following people contributed to this release.

* Stuart Archibald (core dev)


Version 0.38.0
--------------

Following on from the bug fix focus of the last release, this release swings
back towards the addition of new features and usability improvements based on
community feedback. This release is comparatively large! Three key features/
changes to note are:

 * Numba (via llvmlite) is now backed by LLVM 6.0, general vectorization is
   improved as a result. A significant long standing LLVM bug that was causing
   corruption was also found and fixed.
 * Further considerable improvements in vectorization are made available as
   Numba now supports Intel's short vector math library (SVML).
   Try it out with `conda install -c numba icc_rt`.
 * CUDA 8.0 is now the minimum supported CUDA version.

Other highlights include:

 * Bug fixes to `parallel=True` have enabled more vectorization opportunities
   when using the ParallelAccelerator technology.
 * Much effort has gone into improving error reporting and the general usability
   of Numba. This includes highlighted error messages and performance tips
   documentation. Try it out with `conda install colorama`.
 * A number of new NumPy functions are supported, `np.convolve`, `np.correlate`
   `np.reshape`, `np.transpose`, `np.permutation`, `np.real`, `np.imag`, and
   `np.searchsorted` now supports the`side` kwarg. Further, `np.argsort` now
   supports the `kind` kwarg with `quicksort` and `mergesort` available.
 * The Numba extension API has gained the ability operate more easily with
   functions from Cython modules through the use of
   `numba.extending.get_cython_function_address` to obtain function addresses
   for direct use in `ctypes.CFUNCTYPE`.
 * Numba now allows the passing of jitted functions (and containers of jitted
   functions) as arguments to other jitted functions.
 * The CUDA functionality has gained support for a larger selection of bit
   manipulation intrinsics, also SELP, and has had a number of bugs fixed.
 * Initial work to support the PPC64LE platform has been added, full support is
   however waiting on the LLVM 6.0.1 release as it contains critical patches
   not present in 6.0.0.
   It is hoped that any remaining issues will be fixed in the next release.
 * The capacity for advanced users/compiler engineers to define their own
   compilation pipelines.

Enhancements:

* PR #2660: Support bools from cffi in nopython.
* PR #2741: Enhance error message for undefined variables.
* PR #2744: Add diagnostic error message to test suite discovery failure.
* PR #2748: Added Intel SVML optimizations as opt-out choice working by default
* PR #2762: Support transpose with axes arguments.
* PR #2777: Add support for np.correlate and np.convolve
* PR #2779: Implement np.random.permutation
* PR #2801: Passing jitted functions as args
* PR #2802: Support np.real() and np.imag()
* PR #2807: Expose `import_cython_function`
* PR #2821: Add kwarg 'side' to np.searchsorted
* PR #2822: Adds stable argsort
* PR #2832: Fixups for llvmlite 0.23/llvm 6
* PR #2836: Support `index` method on tuples
* PR #2839: Support for np.transpose and np.reshape.
* PR #2843: Custom pipeline
* PR #2847: Replace signed array access indices in unsiged prange loop body
* PR #2859: Add support for improved error reporting.
* PR #2880: This adds a github issue template.
* PR #2881: Build recipe to clone Intel ICC runtime.
* PR #2882: Update TravisCI to test SVML
* PR #2893: Add reference to the data buffer in array.ctypes object
* PR #2895: Move to CUDA 8.0

Fixes:

* PR #2737: Fix #2007 (part 1). Empty array handling in np.linalg.
* PR #2738: Fix install_requires to allow pip getting pre-release version
* PR #2740: Fix 2208. Generate better error message.
* PR #2765: Fix Bit-ness
* PR #2780: PowerPC reference counting memory fences
* PR #2805: Fix six imports.
* PR #2813: Fix #2812: gufunc scalar output bug.
* PR #2814: Fix the build post #2727
* PR #2831: Attempt to fix #2473
* PR #2842: Fix issue with test discovery and broken CUDA drivers.
* PR #2850: Add rtsys init guard and test.
* PR #2852: Skip vectorization test with targets that are not x86
* PR #2856: Prevent printing to stdout in `test_extending.py`
* PR #2864: Correct C code to prevent compiler warnings.
* PR #2889: Attempt to fix #2386.
* PR #2891: Removed test skipping for inspect_cfg
* PR #2898: Add guard to parallel test on unsupported platforms
* PR #2907: Update change log for PPC64LE LLVM dependency.
* PR #2911: Move build requirement to llvmlite>=0.23.0dev0
* PR #2912: Fix random permutation test.
* PR #2914: Fix MD list syntax in issue template.

Documentation Updates:

* PR #2739: Explicitly state default value of error_model in docstring
* PR #2803: DOC: parallel vectorize requires signatures
* PR #2829: Add Python 2.7 EOL plan to docs
* PR #2838: Use automatic numbering syntax in list.
* PR #2877: Add performance tips documentation.
* PR #2883: Fix #2872: update rng doc about thread/fork-safety
* PR #2908: Add missing link and ref to docs.
* PR #2909: Tiny typo correction

ParallelAccelerator enhancements/fixes:

* PR #2727: Changes to enable vectorization in ParallelAccelerator.
* PR #2816: Array analysis for transpose with arbitrary arguments
* PR #2874: Fix dead code eliminator not to remove a call with side-effect
* PR #2886: Fix ParallelAccelerator arrayexpr repr

CUDA enhancements:

* PR #2734: More Constants From cuda.h
* PR #2767: Add len(..) Support to DeviceNDArray
* PR #2778: Add More Device Array API Functions to CUDA Simulator
* PR #2824: Add CUDA Primitives for Population Count
* PR #2835: Emit selp Instructions to Avoid Branching
* PR #2867: Full support for CUDA device attributes

CUDA fixes:
* PR #2768: Don't Compile Code on Every Assignment
* PR #2878: Fixes a Win64 issue with the test in Pr/2865

Contributors:

The following people contributed to this release.

* Abutalib Aghayev
* Alex Olivas
* Anton Malakhov
* Dong-hee Na
* Ehsan Totoni (core dev)
* John Zwinck
* Josh Wilson
* Kelsey Jordahl
* Nick White
* Olexa Bilaniuk
* Rik-de-Kort
* Siu Kwan Lam (core dev)
* Stan Seibert (core dev)
* Stuart Archibald (core dev)
* Thomas Arildsen
* Todd A. Anderson (core dev)


Version 0.37.0
--------------

This release focuses on bug fixing and stability but also adds a few new
features including support for Numpy 1.14. The key change for Numba core was the
long awaited addition of the final tranche of thread safety improvements that
allow Numba to be run concurrently on multiple threads without hitting known
thread safety issues inside LLVM itself. Further, a number of fixes and
enhancements went into the CUDA implementation and ParallelAccelerator gained
some new features and underwent some internal refactoring.

Misc enhancements:

* PR #2627: Remove hacks to make llvmlite threadsafe
* PR #2672: Add ascontiguousarray
* PR #2678: Add Gitter badge
* PR #2691: Fix #2690: add intrinsic to convert array to tuple
* PR #2703: Test runner feature: failed-first and last-failed
* PR #2708: Patch for issue #1907
* PR #2732: Add support for array.fill

Misc Fixes:

* PR #2610: Fix #2606 lowering of optional.setattr
* PR #2650: Remove skip for win32 cosine test
* PR #2668: Fix empty_like from readonly arrays.
* PR #2682: Fixes 2210, remove _DisableJitWrapper
* PR #2684: Fix #2340, generator error yielding bool
* PR #2693: Add travis-ci testing of NumPy 1.14, and also check on Python 2.7
* PR #2694: Avoid type inference failure due to a typing template rejection
* PR #2695: Update llvmlite version dependency.
* PR #2696: Fix tuple indexing codegeneration for empty tuple
* PR #2698: Fix #2697 by deferring deletion in the simplify_CFG loop.
* PR #2701: Small fix to avoid tempfiles being created in the current directory
* PR #2725: Fix 2481, LLVM IR parsing error due to mutated IR
* PR #2726: Fix #2673: incorrect fork error msg.
* PR #2728: Alternative to #2620.  Remove dead code ByteCodeInst.get.
* PR #2730: Add guard for test needing SciPy/BLAS

Documentation updates:

* PR #2670: Update communication channels
* PR #2671: Add docs about diagnosing loop vectorizer
* PR #2683: Add docs on const arg requirements and on const mem alloc
* PR #2722: Add docs on numpy support in cuda
* PR #2724: Update doc: warning about unsupported arguments

ParallelAccelerator enhancements/fixes:

Parallel support for `np.arange` and `np.linspace`, also `np.mean`, `np.std`
and `np.var` are added. This was performed as part of a general refactor and
cleanup of the core ParallelAccelerator code.

* PR #2674: Core pa
* PR #2704: Generate Dels after parfor sequential lowering
* PR #2716: Handle matching directly supported functions

CUDA enhancements:

* PR #2665: CUDA DeviceNDArray: Support numpy tranpose API
* PR #2681: Allow Assigning to DeviceNDArrays
* PR #2702: Make DummyArray do High Dimensional Reshapes
* PR #2714: Use CFFI to Reuse Code

CUDA fixes:

* PR #2667: Fix CUDA DeviceNDArray slicing
* PR #2686: Fix #2663: incorrect offset when indexing cuda array.
* PR #2687: Ensure Constructed Stream Bound
* PR #2706: Workaround for unexpected warp divergence due to exception raising
  code
* PR #2707: Fix regression: cuda test submodules not loading properly in
  runtests
* PR #2731: Use more challenging values in slice tests.
* PR #2720: A quick testsuite fix to not run the new cuda testcase in the
  multiprocess pool

Contributors:

The following people contributed to this release.

* Coutinho Menezes Nilo
* Daniel
* Ehsan Totoni
* Nick White
* Paul H. Liu
* Siu Kwan Lam
* Stan Seibert
* Stuart Archibald
* Todd A. Anderson


Version 0.36.2
--------------

This is a bugfix release that provides minor changes to address:

* PR #2645: Avoid CPython bug with ``exec`` in older 2.7.x.
* PR #2652: Add support for CUDA 9.


Version 0.36.1
--------------

This release continues to add new features to the work undertaken in partnership
with Intel on ParallelAccelerator technology. Other changes of note include the
compilation chain being updated to use LLVM 5.0 and the production of conda
packages using conda-build 3 and the new compilers that ship with it.

NOTE: A version 0.36.0 was tagged for internal use but not released.

ParallelAccelerator:

NOTE: The ParallelAccelerator technology is under active development and should
be considered experimental.

New features relating to ParallelAccelerator, from work undertaken with Intel,
include the addition of the `@stencil` decorator for ease of implementation of
stencil-like computations, support for general reductions, and slice and
range fusion for parallel slice/bit-array assignments. Documentation on both the
use and implementation of the above has been added. Further, a new debug
environment variable `NUMBA_DEBUG_ARRAY_OPT_STATS` is made available to give
information about which operators/calls are converted to parallel for-loops.

ParallelAccelerator features:

* PR #2457: Stencil Computations in ParallelAccelerator
* PR #2548: Slice and range fusion, parallelizing bitarray and slice assignment
* PR #2516: Support general reductions in ParallelAccelerator

ParallelAccelerator fixes:

* PR #2540: Fix bug #2537
* PR #2566: Fix issue #2564.
* PR #2599: Fix nested multi-dimensional parfor type inference issue
* PR #2604: Fixes for stencil tests and cmath sin().
* PR #2605: Fixes issue #2603.

Additional features of note:

This release of Numba (and llvmlite) is updated to use LLVM version 5.0 as the
compiler back end, the main change to Numba to support this was the addition of
a custom symbol tracker to avoid the calls to LLVM's `ExecutionEngine` that was
crashing when asking for non-existent symbol addresses. Further, the conda
packages for this release of Numba are built using conda build version 3 and the
new compilers/recipe grammar that are present in that release.

* PR #2568: Update for LLVM 5
* PR #2607: Fixes abort when getting address to "nrt_unresolved_abort"
* PR #2615: Working towards conda build 3

Thanks to community feedback and bug reports, the following fixes were also
made.

Misc fixes/enhancements:

* PR #2534: Add tuple support to np.take.
* PR #2551: Rebranding fix
* PR #2552: relative doc links
* PR #2570: Fix issue #2561, handle missing successor on loop exit
* PR #2588: Fix #2555. Disable libpython.so linking on linux
* PR #2601: Update llvmlite version dependency.
* PR #2608: Fix potential cache file collision
* PR #2612: Fix NRT test failure due to increased overhead when running in coverage
* PR #2619: Fix dubious pthread_cond_signal not in lock
* PR #2622: Fix `np.nanmedian` for all NaN case.
* PR #2633: Fix markdown in CONTRIBUTING.md
* PR #2635: Make the dependency on compilers for AOT optional.

CUDA support fixes:

* PR #2523: Fix invalid cuda context in memory transfer calls in another thread
* PR #2575: Use CPU to initialize xoroshiro states for GPU RNG. Fixes #2573
* PR #2581: Fix cuda gufunc mishandling of scalar arg as array and out argument


Version 0.35.0
--------------

This release includes some exciting new features as part of the work
performed in partnership with Intel on ParallelAccelerator technology.
There are also some additions made to Numpy support and small but
significant fixes made as a result of considerable effort spent chasing bugs
and implementing stability improvements.


ParallelAccelerator:

NOTE: The ParallelAccelerator technology is under active development and should
be considered experimental.

New features relating to ParallelAccelerator, from work undertaken with Intel,
include support for a larger range of `np.random` functions in `parallel`
mode, printing Numpy arrays in no Python mode, the capacity to initialize Numpy
arrays directly from list comprehensions, and the axis argument to `.sum()`.
Documentation on the ParallelAccelerator technology implementation has also
been added. Further, a large amount of work on equivalence relations was
undertaken to enable runtime checks of broadcasting behaviours in parallel mode.

ParallelAccelerator features:

* PR #2400: Array comprehension
* PR #2405: Support printing Numpy arrays
* PR #2438: from Support more np.random functions in ParallelAccelerator
* PR #2482: Support for sum with axis in nopython mode.
* PR #2487: Adding developer documentation for ParallelAccelerator technology.
* PR #2492: Core PA refactor adds assertions for broadcast semantics

ParallelAccelerator fixes:

* PR #2478: Rename cfg before parfor translation (#2477)
* PR #2479: Fix broken array comprehension tests on unsupported platforms
* PR #2484: Fix array comprehension test on win64
* PR #2506: Fix for 32-bit machines.


Additional features of note:

Support for `np.take`, `np.finfo`, `np.iinfo` and `np.MachAr` in no Python
mode is added. Further, three new environment variables are added, two for
overriding CPU target/features and another to warn if `parallel=True` was set
no such transform was possible.

* PR #2490: Implement np.take and ndarray.take
* PR #2493: Display a warning if parallel=True is set but not possible.
* PR #2513: Add np.MachAr, np.finfo, np.iinfo
* PR #2515: Allow environ overriding of cpu target and cpu features.


Due to expansion of the test farm and a focus on fixing bugs, the following
fixes were also made.

Misc fixes/enhancements:

* PR #2455: add contextual information to runtime errors
* PR #2470: Fixes #2458, poor performance in np.median
* PR #2471: Ensure LLVM threadsafety in {g,}ufunc building.
* PR #2494: Update doc theme
* PR #2503: Remove hacky code added in 2482 and feature enhancement
* PR #2505: Serialise env mutation tests during multithreaded testing.
* PR #2520: Fix failing cpu-target override tests

CUDA support fixes:

* PR #2504: Enable CUDA toolkit version testing
* PR #2509: Disable tests generating code unavailable in lower CC versions.
* PR #2511: Fix Windows 64 bit CUDA tests.


Version 0.34.0
--------------

This release adds a significant set of new features arising from combined work
with Intel on ParallelAccelerator technology. It also adds list comprehension
and closure support, support for Numpy 1.13 and a new, faster, CUDA reduction
algorithm. For Linux users this release is the first to be built on Centos 6,
which will be the new base platform for future releases. Finally a number of
thread-safety, type inference and other smaller enhancements and bugs have been
fixed.


ParallelAccelerator features:

NOTE: The ParallelAccelerator technology is under active development and should
be considered experimental.

The ParallelAccelerator technology is accessed via a new "nopython" mode option
"parallel". The ParallelAccelerator technology attempts to identify operations
which have parallel semantics (for instance adding a scalar to a vector), fuse
together adjacent such operations, and then parallelize their execution across
a number of CPU cores. This is essentially auto-parallelization.

In addition to the auto-parallelization feature, explicit loop based
parallelism is made available through the use of `prange` in place of `range`
as a loop iterator.

More information and examples on both auto-parallelization and `prange` are
available in the documentation and examples directory respectively.

As part of the necessary work for ParallelAccelerator, support for closures
and list comprehensions is added:

* PR #2318: Transfer ParallelAccelerator technology to Numba
* PR #2379: ParallelAccelerator Core Improvements
* PR #2367: Add support for len(range(...))
* PR #2369: List comprehension
* PR #2391: Explicit Parallel Loop Support (prange)

The ParallelAccelerator features are available on all supported platforms and
Python versions with the exceptions of (with view of supporting in a future
release):

* The combination of Windows operating systems with Python 2.7.
* Systems running 32 bit Python.


CUDA support enhancements:

* PR #2377: New GPU reduction algorithm


CUDA support fixes:

* PR #2397: Fix #2393, always set alignment of cuda static memory regions


Misc Fixes:

* PR #2373, Issue #2372: 32-bit compatibility fix for parfor related code
* PR #2376: Fix #2375 missing stdint.h for py2.7 vc9
* PR #2378: Fix deadlock in parallel gufunc when kernel acquires the GIL.
* PR #2382: Forbid unsafe casting in bitwise operation
* PR #2385: docs: fix Sphinx errors
* PR #2396: Use 64-bit RHS operand for shift
* PR #2404: Fix threadsafety logic issue in ufunc compilation cache.
* PR #2424: Ensure consistent iteration order of blocks for type inference.
* PR #2425: Guard code to prevent the use of 'parallel' on win32 + py27
* PR #2426: Basic test for Enum member type recovery.
* PR #2433: Fix up the parfors tests with respect to windows py2.7
* PR #2442: Skip tests that need BLAS/LAPACK if scipy is not available.
* PR #2444: Add test for invalid array setitem
* PR #2449: Make the runtime initialiser threadsafe
* PR #2452: Skip CFG test on 64bit windows


Misc Enhancements:

* PR #2366: Improvements to IR utils
* PR #2388: Update README.rst to indicate the proper version of LLVM
* PR #2394: Upgrade to llvmlite 0.19.*
* PR #2395: Update llvmlite version to 0.19
* PR #2406: Expose environment object to ufuncs
* PR #2407: Expose environment object to target-context inside lowerer
* PR #2413: Add flags to pass through to conda build for buildbot
* PR #2414: Add cross compile flags to local recipe
* PR #2415: A few cleanups for rewrites
* PR #2418: Add getitem support for Enum classes
* PR #2419: Add support for returning enums in vectorize
* PR #2421: Add copyright notice for Intel contributed files.
* PR #2422: Patch code base to work with np 1.13 release
* PR #2448: Adds in warning message when using 'parallel' if cache=True
* PR #2450: Add test for keyword arg on .sum-like and .cumsum-like array
  methods


Version 0.33.0
--------------

This release resolved several performance issues caused by atomic
reference counting operations inside loop bodies.  New optimization
passes have been added to reduce the impact of these operations.  We
observe speed improvements between 2x-10x in affected programs due to
the removal of unnecessary reference counting operations.

There are also several enhancements to the CUDA GPU support:

* A GPU random number generator based on `xoroshiro128+ algorithm <http://xoroshiro.di.unimi.it/>`_ is added.
  See details and examples in :ref:`documentation <cuda-random>`.
* ``@cuda.jit`` CUDA kernels can now call ``@jit`` and ``@njit``
  CPU functions and they will automatically be compiled as CUDA device
  functions.
* CUDA IPC memory API is exposed for sharing memory between proceses.
  See usage details in :ref:`documentation <cuda-ipc-memory>`.

Reference counting enhancements:

* PR #2346, Issue #2345, #2248: Add extra refcount pruning after inlining
* PR #2349: Fix refct pruning not removing refct op with tail call.
* PR #2352, Issue #2350: Add refcount pruning pass for function that does not need refcount

CUDA support enhancements:

* PR #2023: Supports CUDA IPC for device array
* PR #2343, Issue #2335: Allow CPU jit decorated function to be used as cuda device function
* PR #2347: Add random number generator support for CUDA device code
* PR #2361: Update autotune table for CC: 5.3, 6.0, 6.1, 6.2

Misc fixes:

* PR #2362: Avoid test failure due to typing to int32 on 32-bit platforms
* PR #2359: Fixed nogil example that threw a TypeError when executed.
* PR #2357, Issue #2356: Fix fragile test that depends on how the script is executed.
* PR #2355: Fix cpu dispatcher referenced as attribute of another module
* PR #2354: Fixes an issue with caching when function needs NRT and refcount pruning
* PR #2342, Issue #2339: Add warnings to inspection when it is used on unserialized cached code
* PR #2329, Issue #2250: Better handling of missing op codes

Misc enhancements:

* PR #2360: Adds missing values in error mesasge interp.
* PR #2353: Handle when get_host_cpu_features() raises RuntimeError
* PR #2351: Enable SVML for erf/erfc/gamma/lgamma/log2
* PR #2344: Expose error_model setting in jit decorator
* PR #2337: Align blocking terminate support for fork() with new TBB version
* PR #2336: Bump llvmlite version to 0.18
* PR #2330: Core changes in PR #2318


Version 0.32.0
--------------

In this release, we are upgrading to LLVM 4.0.  A lot of work has been done
to fix many race-condition issues inside LLVM when the compiler is
used concurrently, which is likely when Numba is used with Dask.

Improvements:

* PR #2322: Suppress test error due to unknown but consistent error with tgamma
* PR #2320: Update llvmlite dependency to 0.17
* PR #2308: Add details to error message on why cuda support is disabled.
* PR #2302: Add os x to travis
* PR #2294: Disable remove_module on MCJIT due to memory leak inside LLVM
* PR #2291: Split parallel tests and recycle workers to tame memory usage
* PR #2253: Remove the pointer-stuffing hack for storing meminfos in lists

Fixes:

* PR #2331: Fix a bug in the GPU array indexing
* PR #2326: Fix #2321 docs referring to non-existing function.
* PR #2316: Fixing more race-condition problems
* PR #2315: Fix #2314.  Relax strict type check to allow optional type.
* PR #2310: Fix race condition due to concurrent compilation and cache loading
* PR #2304: Fix intrinsic 1st arg not a typing.Context as stated by the docs.
* PR #2287: Fix int64 atomic min-max
* PR #2286: Fix #2285 `@overload_method` not linking dependent libs
* PR #2303: Missing import statements to interval-example.rst


Version 0.31.0
--------------

In this release, we added preliminary support for debugging with GDB
version >= 7.0. The feature is enabled by setting the ``debug=True`` compiler
option, which causes GDB compatible debug info to be generated.
The CUDA backend also gained limited debugging support so that source locations
are showed in memory-checking and profiling tools.
For details, see :ref:`numba-troubleshooting`.

Also, we added the ``fastmath=True`` compiler option to enable unsafe
floating-point transformations, which allows LLVM to auto-vectorize more code.

Other important changes include upgrading to LLVM 3.9.1 and adding support for
Numpy 1.12.

Improvements:

* PR #2281: Update for numpy1.12
* PR #2278: Add CUDA atomic.{max, min, compare_and_swap}
* PR #2277: Add about section to conda recipies to identify license and other
  metadata in Anaconda Cloud
* PR #2271: Adopt itanium C++-style mangling for CPU and CUDA targets
* PR #2267: Add fastmath flags
* PR #2261: Support dtype.type
* PR #2249: Changes for llvm3.9
* PR #2234: Bump llvmlite requirement to 0.16 and add install_name_tool_fixer to
  mviewbuf for OS X
* PR #2230: Add python3.6 to TravisCi
* PR #2227: Enable caching for gufunc wrapper
* PR #2170: Add debugging support
* PR #2037: inspect_cfg() for easier visualization of the function operation

Fixes:

* PR #2274: Fix nvvm ir patch in mishandling "load"
* PR #2272: Fix breakage to cuda7.5
* PR #2269: Fix caching of copy_strides kernel in cuda.reduce
* PR #2265: Fix #2263: error when linking two modules with dynamic globals
* PR #2252: Fix path separator in test
* PR #2246: Fix overuse of memory in some system with fork
* PR #2241: Fix #2240: __module__ in dynamically created function not a str
* PR #2239: Fix fingerprint computation failure preventing fallback


Version 0.30.1
--------------

This is a bug-fix release to enable Python 3.6 support.  In addition,
there is now early Intel TBB support for parallel ufuncs when building from
source with TBBROOT defined.  The TBB feature is not enabled in our official
builds.

Fixes:

* PR #2232: Fix name clashes with _Py_hashtable_xxx in Python 3.6.

Improvements:

* PR #2217: Add Intel TBB threadpool implementation for parallel ufunc.


Version 0.30.0
--------------

This release adds preliminary support for Python 3.6, but no official build is
available yet.  A new system reporting tool (``numba --sysinfo``) is added to
provide system information to help core developers in replication and debugging.
See below for other improvements and bug fixes.

Improvements:

* PR #2209: Support Python 3.6.
* PR #2175: Support ``np.trace()``, ``np.outer()`` and ``np.kron()``.
* PR #2197: Support ``np.nanprod()``.
* PR #2190: Support caching for ufunc.
* PR #2186: Add system reporting tool.

Fixes:

* PR #2214, Issue #2212: Fix memory error with ndenumerate and flat iterators.
* PR #2206, Issue #2163: Fix ``zip()`` consuming extra elements in early
  exhaustion.
* PR #2185, Issue #2159, #2169: Fix rewrite pass affecting objmode fallback.
* PR #2204, Issue #2178: Fix annotation for liftedloop.
* PR #2203: Fix Appveyor segfault with Python 3.5.
* PR #2202, Issue #2198: Fix target context not initialized when loading from
  ufunc cache.
* PR #2172, Issue #2171: Fix optional type unpacking.
* PR #2189, Issue #2188: Disable freezing of big (>1MB) global arrays.
* PR #2180, Issue #2179: Fix invalid variable version in looplifting.
* PR #2156, Issue #2155: Fix divmod, floordiv segfault on CUDA.


Version 0.29.0
--------------

This release extends the support of recursive functions to include direct and
indirect recursion without explicit function type annotations.  See new example
in `examples/mergesort.py`.  Newly supported numpy features include array
stacking functions, np.linalg.eig* functions, np.linalg.matrix_power, np.roots
and array to array broadcasting in assignments.

This release depends on llvmlite 0.14.0 and supports CUDA 8 but it is not
required.

Improvements:

* PR #2130, #2137: Add type-inferred recursion with docs and examples.
* PR #2134: Add ``np.linalg.matrix_power``.
* PR #2125: Add ``np.roots``.
* PR #2129: Add ``np.linalg.{eigvals,eigh,eigvalsh}``.
* PR #2126: Add array-to-array broadcasting.
* PR #2069: Add hstack and related functions.
* PR #2128: Allow for vectorizing a jitted function. (thanks to @dhirschfeld)
* PR #2117: Update examples and make them test-able.
* PR #2127: Refactor interpreter class and its results.

Fixes:

* PR #2149: Workaround MSVC9.0 SP1 fmod bug kb982107.
* PR #2145, Issue #2009: Fixes kwargs for jitclass ``__init__`` method.
* PR #2150: Fix slowdown in objmode fallback.
* PR #2050, Issue #1259: Fix liveness problem with some generator loops.
* PR #2072, Issue #1995: Right shift of unsigned LHS should be logical.
* PR #2115, Issue #1466: Fix inspect_types() error due to mangled variable name.
* PR #2119, Issue #2118: Fix array type created from record-dtype.
* PR #2122, Issue #1808: Fix returning a generator due to datamodel error.


Version 0.28.1
--------------

This is a bug-fix release to resolve packaging issues with setuptools
dependency.


Version 0.28.0
--------------

Amongst other improvements, this version improves again the level of
support for linear algebra -- functions from the :mod:`numpy.linalg`
module.  Also, our random generator is now guaranteed to be thread-safe
and fork-safe.

Improvements:

* PR #2019: Add the ``@intrinsic`` decorator to define low-level
  subroutines callable from JIT functions (this is considered
  a private API for now).
* PR #2059: Implement ``np.concatenate`` and ``np.stack``.
* PR #2048: Make random generation fork-safe and thread-safe, producing
  independent streams of random numbers for each thread or process.
* PR #2031: Add documentation of floating-point pitfalls.
* Issue #2053: Avoid polling in parallel CPU target (fixes severe performance
  regression on Windows).
* Issue #2029: Make default arguments fast.
* PR #2052: Add logging to the CUDA driver.
* PR #2049: Implement the built-in ``divmod()`` function.
* PR #2036: Implement the ``argsort()`` method on arrays.
* PR #2046: Improving CUDA memory management by deferring deallocations
  until certain thresholds are reached, so as to avoid breaking asynchronous
  execution.
* PR #2040: Switch the CUDA driver implementation to use CUDA's
  "primary context" API.
* PR #2017: Allow ``min(tuple)`` and ``max(tuple)``.
* PR #2039: Reduce fork() detection overhead in CUDA.
* PR #2021: Handle structured dtypes with titles.
* PR #1996: Rewrite looplifting as a transformation on Numba IR.
* PR #2014: Implement ``np.linalg.matrix_rank``.
* PR #2012: Implement ``np.linalg.cond``.
* PR #1985: Rewrite even trivial array expressions, which opens the door
  for other optimizations (for example, ``array ** 2`` can be converted
  into ``array * array``).
* PR #1950: Have ``typeof()`` always raise ValueError on failure.
  Previously, it would either raise or return None, depending on the input.
* PR #1994: Implement ``np.linalg.norm``.
* PR #1987: Implement ``np.linalg.det`` and ``np.linalg.slogdet``.
* Issue #1979: Document integer width inference and how to workaround.
* PR #1938: Numba is now compatible with LLVM 3.8.
* PR #1967: Restrict ``np.linalg`` functions to homogeneous dtypes.  Users
  wanting to pass mixed-typed inputs have to convert explicitly, which
  makes the performance implications more obvious.

Fixes:

* PR #2006: ``array(float32) ** int`` should return ``array(float32)``.
* PR #2044: Allow reshaping empty arrays.
* Issue #2051: Fix refcounting issue when concatenating tuples.
* Issue #2000: Make Numpy optional for setup.py, to allow ``pip install``
  to work without Numpy pre-installed.
* PR #1989: Fix assertion in ``Dispatcher.disable_compile()``.
* Issue #2028: Ignore filesystem errors when caching from multiple processes.
* Issue #2003: Allow unicode variable and function names (on Python 3).
* Issue #1998: Fix deadlock in parallel ufuncs that reacquire the GIL.
* PR #1997: Fix random crashes when AOT compiling on certain Windows platforms.
* Issue #1988: Propagate jitclass docstring.
* Issue #1933: Ensure array constants are emitted with the right alignment.


Version 0.27.0
--------------

Improvements:

* Issue #1976: improve error message when non-integral dimensions are given
  to a CUDA kernel.
* PR #1970: Optimize the power operator with a static exponent.
* PR #1710: Improve contextual information for compiler errors.
* PR #1961: Support printing constant strings.
* PR #1959: Support more types in the print() function.
* PR #1823: Support ``compute_50`` in CUDA backend.
* PR #1955: Support ``np.linalg.pinv``.
* PR #1896: Improve the ``SmartArray`` API.
* PR #1947: Support ``np.linalg.solve``.
* Issue #1943: Improve error message when an argument fails typing.4
* PR #1927: Support ``np.linalg.lstsq``.
* PR #1934: Use system functions for hypot() where possible, instead of our
  own implementation.
* PR #1929: Add cffi support to ``@cfunc`` objects.
* PR #1932: Add user-controllable thread pool limits for parallel CPU target.
* PR #1928: Support self-recursion when the signature is explicit.
* PR #1890: List all lowering implementations in the developer docs.
* Issue #1884: Support ``np.lib.stride_tricks.as_strided()``.

Fixes:

* Issue #1960: Fix sliced assignment when source and destination areas are
  overlapping.
* PR #1963: Make CUDA print() atomic.
* PR #1956: Allow 0d array constants.
* Issue #1945: Allow using Numpy ufuncs in AOT compiled code.
* Issue #1916: Fix documentation example for ``@generated_jit``.
* Issue #1926: Fix regression when caching functions in an IPython session.
* Issue #1923: Allow non-intp integer arguments to carray() and farray().
* Issue #1908: Accept non-ASCII unicode docstrings on Python 2.
* Issue #1874: Allow ``del container[key]`` in object mode.
* Issue #1913: Fix set insertion bug when the lookup chain contains deleted
  entries.
* Issue #1911: Allow function annotations on jitclass methods.


Version 0.26.0
--------------

This release adds support for ``cfunc`` decorator for exporting numba jitted
functions to 3rd party API that takes C callbacks.  Most of the overhead of
using jitclasses inside the interpreter are eliminated.  Support for
decompositions in ``numpy.linalg`` are added.  Finally, Numpy 1.11 is
supported.

Improvements:

* PR #1889: Export BLAS and LAPACK wrappers for pycc.
* PR #1888: Faster array power.
* Issue #1867: Allow "out" keyword arg for dufuncs.
* PR #1871: ``carray()`` and ``farray()`` for creating arrays from pointers.
* PR #1855: ``@cfunc`` decorator for exporting as ctypes function.
* PR #1862: Add support for ``numpy.linalg.qr``.
* PR #1851: jitclass support for '_' and '__' prefixed attributes.
* PR #1842: Optimize jitclass in Python interpreter.
* Issue #1837: Fix CUDA simulator issues with device function.
* PR #1839: Add support for decompositions from ``numpy.linalg``.
* PR #1829: Support Python enums.
* PR #1828: Add support for ``numpy.random.rand()``` and
  ``numpy.random.randn()``
* Issue #1825: Use of 0-darray in place of scalar index.
* Issue #1824: Scalar arguments to object mode gufuncs.
* Issue #1813: Let bitwise bool operators return booleans, not integers.
* Issue #1760: Optional arguments in generators.
* PR #1780: Numpy 1.11 support.


Version 0.25.0
--------------

This release adds support for ``set`` objects in nopython mode.  It also
adds support for many missing Numpy features and functions.  It improves
Numba's compatibility and performance when using a distributed execution
framework such as dask, distributed or Spark.  Finally, it removes
compatibility with Python 2.6, Python 3.3 and Numpy 1.6.

Improvements:

* Issue #1800: Add erf(), erfc(), gamma() and lgamma() to CUDA targets.
* PR #1793: Implement more Numpy functions: np.bincount(), np.diff(),
  np.digitize(), np.histogram(), np.searchsorted() as well as NaN-aware
  reduction functions (np.nansum(), np.nanmedian(), etc.)
* PR #1789: Optimize some reduction functions such as np.sum(), np.prod(),
  np.median(), etc.
* PR #1752: Make CUDA features work in dask, distributed and Spark.
* PR #1787: Support np.nditer() for fast multi-array indexing with
  broadcasting.
* PR #1799: Report JIT-compiled functions as regular Python functions
  when profiling (allowing to see the filename and line number where a
  function is defined).
* PR #1782: Support np.any() and np.all().
* Issue #1788: Support the iter() and next() built-in functions.
* PR #1778: Support array.astype().
* Issue #1775: Allow the user to set the target CPU model for AOT compilation.
* PR #1758: Support creating random arrays using the ``size`` parameter
  to the np.random APIs.
* PR #1757: Support len() on array.flat objects.
* PR #1749: Remove Numpy 1.6 compatibility.
* PR #1748: Remove Python 2.6 and 3.3 compatibility.
* PR #1735: Support the ``not in`` operator as well as operator.contains().
* PR #1724: Support homogeneous sets in nopython mode.
* Issue #875: make compilation of array constants faster.

Fixes:

* PR #1795: Fix a massive performance issue when calling Numba functions
  with distributed, Spark or a similar mechanism using serialization.
* Issue #1784: Make jitclasses usable with NUMBA_DISABLE_JIT=1.
* Issue #1786: Allow using linear algebra functions when profiling.
* Issue #1796: Fix np.dot() memory leak on non-contiguous inputs.
* PR #1792: Fix static negative indexing of tuples.
* Issue #1771: Use fallback cache directory when __pycache__ isn't writable,
  such as when user code is installed in a system location.
* Issue #1223: Use Numpy error model in array expressions (e.g. division
  by zero returns ``inf`` or ``nan`` instead of raising an error).
* Issue #1640: Fix np.random.binomial() for large n values.
* Issue #1643: Improve error reporting when passing an invalid spec to
  ``@jitclass``.
* PR #1756: Fix slicing with a negative step and an omitted start.


Version 0.24.0
--------------

This release introduces several major changes, including the ``@generated_jit``
decorator for flexible specializations as with Julia's "``@generated``" macro,
or the SmartArray array wrapper type that allows seamless transfer of array
data between the CPU and the GPU.

This will be the last version to support Python 2.6, Python 3.3 and Numpy 1.6.

Improvements:

* PR #1723: Improve compatibility of JIT functions with the Python profiler.
* PR #1509: Support array.ravel() and array.flatten().
* PR #1676: Add SmartArray type to support transparent data management in
  multiple address spaces (host & GPU).
* PR #1689: Reduce startup overhead of importing Numba.
* PR #1705: Support registration of CFFI types as corresponding to known
  Numba types.
* PR #1686: Document the extension API.
* PR #1698: Improve warnings raised during type inference.
* PR #1697: Support np.dot() and friends on non-contiguous arrays.
* PR #1692: cffi.from_buffer() improvements (allow more pointer types,
  allow non-Numpy buffer objects).
* PR #1648: Add the ``@generated_jit`` decorator.
* PR #1651: Implementation of np.linalg.inv using LAPACK.  Thanks to
  Matthieu Dartiailh.
* PR #1674: Support np.diag().
* PR #1673: Improve error message when looking up an attribute on an
  unknown global.
* Issue #1569: Implement runtime check for the LLVM locale bug.
* PR #1612: Switch to LLVM 3.7 in sync with llvmlite.
* PR #1624: Allow slice assignment of sequence to array.
* PR #1622: Support slicing tuples with a constant slice.

Fixes:

* Issue #1722: Fix returning an optional boolean (bool or None).
* Issue #1734: NRT decref bug when variable is del'ed before being defined,
  leading to a possible memory leak.
* PR #1732: Fix tuple getitem regression for CUDA target.
* PR #1718: Mishandling of optional to optional casting.
* PR #1714: Fix .compile() on a JIT function not respecting ._can_compile.
* Issue #1667: Fix np.angle() on arrays.
* Issue #1690: Fix slicing with an omitted stop and a negative step value.
* PR #1693: Fix gufunc bug in handling scalar formal arg with non-scalar
  input value.
* PR #1683: Fix parallel testing under Windows.
* Issue #1616: Use system-provided versions of C99 math where possible.
* Issue #1652: Reductions of bool arrays (e.g. sum() or mean()) should
  return integers or floats, not bools.
* Issue #1664: Fix regression when indexing a record array with a constant
  index.
* PR #1661: Disable AVX on old Linux kernels.
* Issue #1636: Allow raising an exception looked up on a module.


Version 0.23.1
--------------

This is a bug-fix release to address several regressions introduced
in the 0.23.0 release, and a couple other issues.

Fixes:

* Issue #1645: CUDA ufuncs were broken in 0.23.0.
* Issue #1638: Check tuple sizes when passing a list of tuples.
* Issue #1630: Parallel ufunc would keep eating CPU even after finishing
  under Windows.
* Issue #1628: Fix ctypes and cffi tests under Windows with Python 3.5.
* Issue #1627: Fix xrange() support.
* PR #1611: Rewrite variable liveness analysis.
* Issue #1610: Allow nested calls between explicitly-typed ufuncs.
* Issue #1593: Fix `*args` in object mode.


Version 0.23.0
--------------

This release introduces JIT classes using the new ``@jitclass`` decorator,
allowing user-defined structures for nopython mode.  Other improvements
and bug fixes are listed below.

Improvements:

* PR #1609: Speed up some simple math functions by inlining them
  in their caller
* PR #1571: Implement JIT classes
* PR #1584: Improve typing of array indexing
* PR #1583: Allow printing booleans
* PR #1542: Allow negative values in np.reshape()
* PR #1560: Support vector and matrix dot product, including ``np.dot()``
  and the ``@`` operator in Python 3.5
* PR #1546: Support field lookup on record arrays and scalars (i.e.
  ``array['field']`` in addition to ``array.field``)
* PR #1440: Support the HSA wavebarrier() and activelanepermute_wavewidth()
  intrinsics
* PR #1540: Support np.angle()
* PR #1543: Implement CPU multithreaded gufuncs (target="parallel")
* PR #1551: Allow scalar arguments in np.where(), np.empty_like().
* PR #1516: Add some more examples from NumbaPro
* PR #1517: Support np.sinc()

Fixes:

* Issue #1603: Fix calling a non-cached function from a cached function
* Issue #1594: Ensure a list is homogeneous when unboxing
* Issue #1595: Replace deprecated use of get_pointer_to_function()
* Issue #1586: Allow tests to be run by different users on the same machine
* Issue #1587: Make CudaAPIError picklable
* Issue #1568: Fix using Numba from inside Visual Studio 2015
* Issue #1559: Fix serializing a jit function referring a renamed module
* PR #1508: Let reshape() accept integer argument(s), not just a tuple
* Issue #1545: Improve error checking when unboxing list objects
* Issue #1538: Fix array broadcasting in CUDA gufuncs
* Issue #1526: Fix a reference count handling bug


Version 0.22.1
--------------

This is a bug-fix release to resolve some packaging issues and other
problems found in the 0.22.0 release.

Fixes:

* PR #1515: Include MANIFEST.in in MANIFEST.in so that sdist still works from
  source tar files.
* PR #1518: Fix reference counting bug caused by hidden alias
* PR #1519: Fix erroneous assert when passing nopython=True to guvectorize.
* PR #1521: Fix cuda.test()

Version 0.22.0
--------------

This release features several highlights: Python 3.5 support, Numpy 1.10
support, Ahead-of-Time compilation of extension modules, additional
vectorization features that were previously only available with the
proprietary extension NumbaPro, improvements in array indexing.

Improvements:

* PR #1497: Allow scalar input type instead of size-1 array to @guvectorize
* PR #1480: Add distutils support for AOT compilation
* PR #1460: Create a new API for Ahead-of-Time (AOT) compilation
* PR #1451: Allow passing Python lists to JIT-compiled functions, and
  reflect mutations on function return
* PR #1387: Numpy 1.10 support
* PR #1464: Support cffi.FFI.from_buffer()
* PR #1437: Propagate errors raised from Numba-compiled ufuncs; also,
  let "division by zero" and other math errors produce a warning instead
  of exiting the function early
* PR #1445: Support a subset of fancy indexing
* PR #1454: Support "out-of-line" CFFI modules
* PR #1442: Improve array indexing to support more kinds of basic slicing
* PR #1409: Support explicit CUDA memory fences
* PR #1435: Add support for vectorize() and guvectorize() with HSA
* PR #1432: Implement numpy.nonzero() and numpy.where()
* PR #1416: Add support for vectorize() and guvectorize() with CUDA,
  as originally provided in NumbaPro
* PR #1424: Support in-place array operators
* PR #1414: Python 3.5 support
* PR #1404: Add the parallel ufunc functionality originally provided in
  NumbaPro
* PR #1393: Implement sorting on arrays and lists
* PR #1415: Add functions to estimate the occupancy of a CUDA kernel
* PR #1360: The JIT cache now stores the compiled object code, yielding
  even larger speedups.
* PR #1402: Fixes for the ARMv7 (armv7l) architecture under Linux
* PR #1400: Add the cuda.reduce() decorator originally provided in NumbaPro

Fixes:

* PR #1483: Allow np.empty_like() and friends on non-contiguous arrays
* Issue #1471: Allow caching JIT functions defined in IPython
* PR #1457: Fix flat indexing of boolean arrays
* PR #1421: Allow calling Numpy ufuncs, without an explicit output, on
  non-contiguous arrays
* Issue #1411: Fix crash when unpacking a tuple containing a Numba-allocated array
* Issue #1394: Allow unifying range_state32 and range_state64
* Issue #1373: Fix code generation error on lists of bools


Version 0.21.0
--------------

This release introduces support for AMD's Heterogeneous System Architecture,
which allows memory to be shared directly between the CPU and the GPU.
Other major enhancements are support for lists and the introduction of
an opt-in compilation cache.

Improvements:

* PR #1391: Implement print() for CUDA code
* PR #1366: Implement integer typing enhancement proposal (NBEP 1)
* PR #1380: Support the one-argument type() builtin
* PR #1375: Allow boolean evaluation of lists and tuples
* PR #1371: Support array.view() in CUDA mode
* PR #1369: Support named tuples in nopython mode
* PR #1250: Implement numpy.median().
* PR #1289: Make dispatching faster when calling a JIT-compiled function
  from regular Python
* Issue #1226: Improve performance of integer power
* PR #1321: Document features supported with CUDA
* PR #1345: HSA support
* PR #1343: Support lists in nopython mode
* PR #1356: Make Numba-allocated memory visible to tracemalloc
* PR #1363: Add an environment variable NUMBA_DEBUG_TYPEINFER
* PR #1051: Add an opt-in, per-function compilation cache

Fixes:

* Issue #1372: Some array expressions would fail rewriting when involved
  the same variable more than once, or a unary operator
* Issue #1385: Allow CUDA local arrays to be declared anywhere in a function
* Issue #1285: Support datetime64 and timedelta64 in Numpy reduction functions
* Issue #1332: Handle the EXTENDED_ARG opcode.
* PR #1329: Handle the ``in`` operator in object mode
* Issue #1322: Fix augmented slice assignment on Python 2
* PR #1357: Fix slicing with some negative bounds or step values.


Version 0.20.0
--------------

This release updates Numba to use LLVM 3.6 and CUDA 7 for CUDA support.
Following the platform deprecation in CUDA 7, Numba's CUDA feature is no
longer supported on 32-bit platforms.  The oldest supported version of
Windows is Windows 7.

Improvements:

* Issue #1203: Support indexing ndarray.flat
* PR #1200: Migrate cgutils to llvmlite
* PR #1190: Support more array methods: .transpose(), .T, .copy(), .reshape(), .view()
* PR #1214: Simplify setup.py and avoid manual maintenance
* PR #1217: Support datetime64 and timedelta64 constants
* PR #1236: Reload environment variables when compiling
* PR #1225: Various speed improvements in generated code
* PR #1252: Support cmath module in CUDA
* PR #1238: Use 32-byte aligned allocator to optimize for AVX
* PR #1258: Support numpy.frombuffer()
* PR #1274: Use TravisCI container infrastructure for lower wait time
* PR #1279: Micro-optimize overload resolution in call dispatch
* Issue #1248: Improve error message when return type unification fails

Fixes:

* Issue #1131: Handling of negative zeros in np.conjugate() and np.arccos()
* Issue #1188: Fix slow array return
* Issue #1164: Avoid warnings from CUDA context at shutdown
* Issue #1229: Respect the writeable flag in arrays
* Issue #1244: Fix bug in refcount pruning pass
* Issue #1251: Fix partial left-indexing of Fortran contiguous array
* Issue #1264: Fix compilation error in array expression
* Issue #1254: Fix error when yielding array objects
* Issue #1276: Fix nested generator use


Version 0.19.2
--------------

This release fixes the source distribution on pypi.  The only change is in the
setup.py file.  We do not plan to provide a conda package as this release is
essentially the same as 0.19.1 for conda users.


Version 0.19.1
--------------

* Issue #1196:

  * fix double-free segfault due to redundant variable deletion in the
    Numba IR (#1195)
  * fix use-after-delete in array expression rewrite pass


Version 0.19.0
--------------

This version introduces memory management in the Numba runtime, allowing to
allocate new arrays inside Numba-compiled functions.  There is also a rework
of the ufunc infrastructure, and an optimization pass to collapse cascading
array operations into a single efficient loop.

.. warning::
   Support for Windows XP and Vista with all compiler targets and support
   for 32-bit platforms (Win/Mac/Linux) with the CUDA compiler target are
   deprecated.  In the next release of Numba, the oldest version of Windows
   supported will be Windows 7.  CPU compilation will remain supported
   on 32-bit Linux and Windows platforms.

Known issues:

* There are some performance regressions in very short running ``nopython``
  functions due to the additional overhead incurred by memory management.
  We will work to reduce this overhead in future releases.

Features:

* Issue #1181: Add a Frequently Asked Questions section to the documentation.
* Issue #1162: Support the ``cumsum()`` and ``cumprod()`` methods on Numpy
  arrays.
* Issue #1152: Support the ``*args`` argument-passing style.
* Issue #1147: Allow passing character sequences as arguments to
  JIT-compiled functions.
* Issue #1110: Shortcut deforestation and loop fusion for array expressions.
* Issue #1136: Support various Numpy array constructors, for example
  numpy.zeros() and numpy.zeros_like().
* Issue #1127: Add a CUDA simulator running on the CPU, enabled with the
  NUMBA_ENABLE_CUDASIM environment variable.
* Issue #1086: Allow calling standard Numpy ufuncs without an explicit
  output array from ``nopython`` functions.
* Issue #1113: Support keyword arguments when calling numpy.empty()
  and related functions.
* Issue #1108: Support the ``ctypes.data`` attribute of Numpy arrays.
* Issue #1077: Memory management for array allocations in ``nopython`` mode.
* Issue #1105: Support calling a ctypes function that takes ctypes.py_object
  parameters.
* Issue #1084: Environment variable NUMBA_DISABLE_JIT disables compilation
  of ``@jit`` functions, instead calling into the Python interpreter
  when called.  This allows easier debugging of multiple jitted functions.
* Issue #927: Allow gufuncs with no output array.
* Issue #1097: Support comparisons between tuples.
* Issue #1075: Numba-generated ufuncs can now be called from ``nopython``
  functions.
* Issue #1062: ``@vectorize`` now allows omitting the signatures, and will
  compile the required specializations on the fly (like ``@jit`` does).
* Issue #1027: Support numpy.round().
* Issue #1085: Allow returning a character sequence (as fetched from a
  structured array) from a JIT-compiled function.

Fixes:

* Issue #1170: Ensure ``ndindex()``, ``ndenumerate()`` and ``ndarray.flat``
  work properly inside generators.
* Issue #1151: Disallow unpacking of tuples with the wrong size.
* Issue #1141: Specify install dependencies in setup.py.
* Issue #1106: Loop-lifting would fail when the lifted loop does not
  produce any output values for the function tail.
* Issue #1103: Fix mishandling of some inputs when a JIT-compiled function
  is called with multiple array layouts.
* Issue #1089: Fix range() with large unsigned integers.
* Issue #1088: Install entry-point scripts (numba, pycc) from the conda
  build recipe.
* Issue #1081: Constant structured scalars now work properly.
* Issue #1080: Fix automatic promotion of booleans to integers.


Version 0.18.2
--------------

Bug fixes:

* Issue #1073: Fixes missing template file for HTML annotation
* Issue #1074: Fixes CUDA support on Windows machine due to NVVM API mismatch


Version 0.18.1
--------------

Version 0.18.0 is not officially released.

This version removes the old deprecated and undocumented ``argtypes`` and
``restype`` arguments to the ``@jit`` decorator.  Function signatures
should always be passed as the first argument to ``@jit``.

Features:

* Issue #960: Add inspect_llvm() and inspect_asm() methods to JIT-compiled
  functions: they output the LLVM IR and the native assembler source of the
  compiled function, respectively.
* Issue #990: Allow passing tuples as arguments to JIT-compiled functions
  in ``nopython`` mode.
* Issue #774: Support two-argument round() in ``nopython`` mode.
* Issue #987: Support missing functions from the math module in nopython
  mode: frexp(), ldexp(), gamma(), lgamma(), erf(), erfc().
* Issue #995: Improve code generation for round() on Python 3.
* Issue #981: Support functions from the random and numpy.random modules
  in ``nopython`` mode.
* Issue #979: Add cuda.atomic.max().
* Issue #1006: Improve exception raising and reporting.  It is now allowed
  to raise an exception with an error message in ``nopython`` mode.
* Issue #821: Allow ctypes- and cffi-defined functions as arguments to
  ``nopython`` functions.
* Issue #901: Allow multiple explicit signatures with ``@jit``.  The
  signatures must be passed in a list, as with ``@vectorize``.
* Issue #884: Better error message when a JIT-compiled function is called
  with the wrong types.
* Issue #1010: Simpler and faster CUDA argument marshalling thanks to a
  refactoring of the data model.
* Issue #1018: Support arrays of scalars inside Numpy structured types.
* Issue #808: Reduce Numba import time by half.
* Issue #1021: Support the buffer protocol in ``nopython`` mode.
  Buffer-providing objects, such as ``bytearray``, ``array.array`` or
  ``memoryview`` support array-like operations such as indexing and iterating.
  Furthermore, some standard attributes on the ``memoryview`` object are
  supported.
* Issue #1030: Support nested arrays in Numpy structured arrays.
* Issue #1033: Implement the inspect_types(), inspect_llvm() and inspect_asm()
  methods for CUDA kernels.
* Issue #1029: Support Numpy structured arrays with CUDA as well.
* Issue #1034: Support for generators in nopython and object mode.
* Issue #1044: Support default argument values when calling Numba-compiled
  functions.
* Issue #1048: Allow calling Numpy scalar constructors from CUDA functions.
* Issue #1047: Allow indexing a multi-dimensional array with a single integer,
  to take a view.
* Issue #1050: Support len() on tuples.
* Issue #1011: Revive HTML annotation.

Fixes:

* Issue #977: Assignment optimization was too aggressive.
* Issue #561: One-argument round() now returns an int on Python 3.
* Issue #1001: Fix an unlikely bug where two closures with the same name
  and id() would compile to the same LLVM function name, despite different
  closure values.
* Issue #1006: Fix reference leak when a JIT-compiled function is disposed of.
* Issue #1017: Update instructions for CUDA in the README.
* Issue #1008: Generate shorter LLVM type names to avoid segfaults with CUDA.
* Issue #1005: Properly clean up references when raising an exception from
  object mode.
* Issue #1041: Fix incompatibility between Numba and the third-party
  library "future".
* Issue #1053: Fix the size attribute of CUDA shared arrays.


Version 0.17.0
--------------

The major focus in this release has been a rewrite of the documentation.
The new documentation is better structured and has more detailed coverage
of Numba features and APIs.  It can be found online at
https://numba.pydata.org/numba-doc/dev/index.html

Features:

* Issue #895: LLVM can now inline nested function calls in ``nopython`` mode.
* Issue #863: CUDA kernels can now infer the types of their arguments
  ("autojit"-like).
* Issue #833: Support numpy.{min,max,argmin,argmax,sum,mean,var,std}
  in ``nopython`` mode.
* Issue #905: Add a ``nogil`` argument to the ``@jit`` decorator, to
  release the GIL in ``nopython`` mode.
* Issue #829: Add a ``identity`` argument to ``@vectorize`` and
  ``@guvectorize``, to set the identity value of the ufunc.
* Issue #843: Allow indexing 0-d arrays with the empty tuple.
* Issue #933: Allow named arguments, not only positional arguments, when
  calling a Numba-compiled function.
* Issue #902: Support numpy.ndenumerate() in ``nopython`` mode.
* Issue #950: AVX is now enabled by default except on Sandy Bridge and
  Ivy Bridge CPUs, where it can produce slower code than SSE.
* Issue #956: Support constant arrays of structured type.
* Issue #959: Indexing arrays with floating-point numbers isn't allowed
  anymore.
* Issue #955: Add support for 3D CUDA grids and thread blocks.
* Issue #902: Support numpy.ndindex() in ``nopython`` mode.
* Issue #951: Numpy number types (``numpy.int8``, etc.) can be used as
  constructors for type conversion in ``nopython`` mode.

Fixes:

* Issue #889: Fix ``NUMBA_DUMP_ASSEMBLY`` for the CUDA backend.
* Issue #903: Fix calling of stdcall functions with ctypes under Windows.
* Issue #908: Allow lazy-compiling from several threads at once.
* Issue #868: Wrong error message when multiplying a scalar by a non-scalar.
* Issue #917: Allow vectorizing with datetime64 and timedelta64 in the
  signature (only with unit-less values, though, because of a Numpy limitation).
* Issue #431: Allow overloading of cuda device function.
* Issue #917: Print out errors occurred in object mode ufuncs.
* Issue #923: Numba-compiled ufuncs now inherit the name and doc of the
  original Python function.
* Issue #928: Fix boolean return value in nested calls.
* Issue #915: ``@jit`` called with an explicit signature with a mismatching
  type of arguments now raises an error.
* Issue #784: Fix the truth value of NaNs.
* Issue #953: Fix using shared memory in more than one function (kernel or
  device).
* Issue #970: Fix an uncommon double to uint64 conversion bug on CentOS5
  32-bit (C compiler issue).


Version 0.16.0
--------------

This release contains a major refactor to switch from llvmpy to `llvmlite <https://github.com/numba/llvmlite>`_
as our code generation backend.  The switch is necessary to reconcile
different compiler requirements for LLVM 3.5 (needs C++11) and Python
extensions (need specific compiler versions on Windows). As a bonus, we have
found the use of llvmlite speeds up compilation by a factor of 2!

Other Major Changes:

* Faster dispatch for numpy structured arrays
* Optimized array.flat()
* Improved CPU feature selection
* Fix constant tuple regression in macro expansion code

Known Issues:

* AVX code generation is still disabled by default due to performance
  regressions when operating on misaligned NumPy arrays.  We hope to have a
  workaround in the future.
* In *extremely* rare circumstances, a `known issue with LLVM 3.5 <http://llvm.org/bugs/show_bug.cgi?id=21423>`_
  code generation can cause an ELF relocation error on 64-bit Linux systems.


Version 0.15.1
--------------

(This was a bug-fix release that superceded version 0.15 before it was
announced.)

Fixes:

* Workaround for missing __ftol2 on Windows XP.
* Do not lift loops for compilation that contain break statements.
* Fix a bug in loop-lifting when multiple values need to be returned to
  the enclosing scope.
* Handle the loop-lifting case where an accumulator needs to be updated when
  the loop count is zero.

Version 0.15
------------

Features:

* Support for the Python ``cmath`` module.  (NumPy complex functions were
  already supported.)
* Support for ``.real``, ``.imag``, and `.conjugate()`` on non-complex
  numbers.
* Add support for ``math.isfinite()`` and ``math.copysign()``.
* Compatibility mode: If enabled (off by default), a failure to compile in
  object mode will fall back to using the pure Python implementation of the
  function.
* *Experimental* support for serializing JIT functions with cloudpickle.
* Loop-jitting in object mode now works with loops that modify scalars that
  are accessed after the loop, such as accumulators.
* ``@vectorize`` functions can be compiled in object mode.
* Numba can now be built using the `Visual C++ Compiler for Python 2.7 <http://aka.ms/vcpython27>`_
  on Windows platforms.
* CUDA JIT functions can be returned by factory functions with variables in
  the closure frozen as constants.
* Support for "optional" types in nopython mode, which allow ``None`` to be a
  valid value.

Fixes:

* If nopython mode compilation fails for any reason, automatically fall back
  to object mode (unless nopython=True is passed to @jit) rather than raise
  an exeception.
* Allow function objects to be returned from a function compiled in object
  mode.
* Fix a linking problem that caused slower platform math functions (such as
  ``exp()``) to be used on Windows, leading to performance regressions against
  NumPy.
* ``min()`` and ``max()`` no longer accept scalars arguments in nopython mode.
* Fix handling of ambigous type promotion among several compiled versions of a
  JIT function.  The dispatcher will now compile a new version to resolve the
  problem.  (issue #776)
* Fix float32 to uint64 casting bug on 32-bit Linux.
* Fix type inference to allow forced casting of return types.
* Allow the shape of a 1D ``cuda.shared.array`` and ``cuda.local.array`` to be
  a one-element tuple.
* More correct handling of signed zeros.
* Add custom implementation of ``atan2()`` on Windows to handle special cases
  properly.
* Eliminated race condition in the handling of the pagelocked staging area
  used when transferring CUDA arrays.
* Fix non-deterministic type unification leading to varying performance.
  (issue #797)


Version 0.14
------------

Features:

* Support for nearly all the Numpy math functions (including comparison,
  logical, bitwise and some previously missing float functions) in nopython mode.
* The Numpy datetime64 and timedelta64 dtypes are supported in nopython mode
  with Numpy 1.7 and later.
* Support for Numpy math functions on complex numbers in nopython mode.
* ndarray.sum() is supported in nopython mode.
* Better error messages when unsupported types are used in Numpy math functions.
* Set NUMBA_WARNINGS=1 in the environment to see which functions are compiled
  in object mode vs. nopython mode.
* Add support for the two-argument pow() builtin function in nopython mode.
* New developer documentation describing how Numba works, and how to
  add new types.
* Support for Numpy record arrays on the GPU. (Note: Improper alignment of dtype
  fields will cause an exception to be raised.)
* Slices on GPU device arrays.
* GPU objects can be used as Python context managers to select the active
  device in a block.
* GPU device arrays can be bound to a CUDA stream.  All subsequent operations
  (such as memory copies) will be queued on that stream instead of the default.
  This can prevent unnecessary synchronization with other streams.

Fixes:

* Generation of AVX instructions has been disabled to avoid performance bugs
  when calling external math functions that may use SSE instructions,
  especially on OS X.
* JIT functions can be removed by the garbage collector when they are no
  longer accessible.
* Various other reference counting fixes to prevent memory leaks.
* Fixed handling of exception when input argument is out of range.
* Prevent autojit functions from making unsafe numeric conversions when
  called with different numeric types.
* Fix a compilation error when an unhashable global value is accessed.
* Gracefully handle failure to enable faulthandler in the IPython Notebook.
* Fix a bug that caused loop lifting to fail if the loop was inside an
  ``else`` block.
* Fixed a problem with selecting CUDA devices in multithreaded programs on
  Linux.
* The ``pow()`` function (and ``**`` operation) applied to two integers now
  returns an integer rather than a float.
* Numpy arrays using the object dtype no longer cause an exception in the
  autojit.
* Attempts to write to a global array will cause compilation to fall back
  to object mode, rather than attempt and fail at nopython mode.
* ``range()`` works with all negative arguments (ex: ``range(-10, -12, -1)``)

Version 0.13.4
--------------

Features:

* Setting and deleting attributes in object mode
* Added documentation of supported and currently unsupported numpy ufuncs
* Assignment to 1-D numpy array slices
* Closure variables and functions can be used in object mode
* All numeric global values in modules can be used as constants in JIT
  compiled code
* Support for the start argument in enumerate()
* Inplace arithmetic operations (+=, -=, etc.)
* Direct iteration over a 1D numpy array (e.g. "for x in array: ...")
  in nopython mode

Fixes:

* Support for NVIDIA compute capability 5.0 devices (such as the GTX 750)
* Vectorize no longer crashes/gives an error when bool\_ is used as return type
* Return the correct dictionary when globals() is used in JIT functions
* Fix crash bug when creating dictionary literals in object
* Report more informative error message on import if llvmpy is too old
* Temporarily disable pycc --header, which generates incorrect function
  signatures.

Version 0.13.3
--------------

Features:

* Support for enumerate() and zip() in nopython mode
* Increased LLVM optimization of JIT functions to -O1, enabling automatic
  vectorization of compiled code in some cases
* Iteration over tuples and unpacking of tuples in nopython mode
* Support for dict and set (Python >= 2.7) literals in object mode

Fixes:

* JIT functions have the same __name__ and __doc__ as the original function.
* Numerous improvements to better match the data types and behavior of Python
  math functions in JIT compiled code on different platforms.
* Importing Numba will no longer throw an exception if the CUDA driver is
  present, but cannot be initialized.
* guvectorize now properly supports functions with scalar arguments.
* CUDA driver is lazily initialized

Version 0.13.2
--------------

Features:

* @vectorize ufunc now can generate SIMD fast path for unit strided array
* Added cuda.gridsize
* Added preliminary exception handling (raise exception class)

Fixes:

* UNARY_POSITIVE
* Handling of closures and dynamically generated functions
* Global None value

Version 0.13.1
--------------

Features:

* Initial support for CUDA array slicing

Fixes:

* Indirectly fixes numbapro when the system has a incompatible CUDA driver
* Fix numba.cuda.detect
* Export numba.intp and numba.intc

Version 0.13
------------

Features:

* Opensourcing NumbaPro CUDA python support in `numba.cuda`
* Add support for ufunc array broadcasting
* Add support for mixed input types for ufuncs
* Add support for returning tuple from jitted function

Fixes:

* Fix store slice bytecode handling for Python2
* Fix inplace subtract
* Fix pycc so that correct header is emitted
* Allow vectorize to work on functions with jit decorator


Version 0.12.2
--------------

Fixes:

* Improved NumPy ufunc support in nopython mode
* Misc bug fixes


Version 0.12.1
--------------

This version fixed many regressions reported by user for the 0.12 release.
This release contains a new loop-lifting mechanism that specializes certains
loop patterns for nopython mode compilation.  This avoid direct support
for heap-allocating and other very dynamic operations.

Improvements:

* Add loop-lifting--jit-ing loops in nopython for object mode code. This allows
  functions to allocate NumPy arrays and use Python objects, while the tight
  loops in the function can still be compiled in nopython mode. Any arrays that
  the tight loop uses should be created before the loop is entered.

Fixes:

* Add support for majority of "math" module functions
* Fix for...else handling
* Add support for builtin round()
* Fix tenary if...else support
* Revive "numba" script
* Fix problems with some boolean expressions
* Add support for more NumPy ufuncs


Version 0.12
------------

Version 0.12 contains a big refactor of the compiler. The main objective for
this refactor was to simplify the code base to create a better foundation for
further work. A secondary objective was to improve the worst case performance
to ensure that compiled functions in object mode never run slower than pure
Python code (this was a problem in several cases with the old code base). This
refactor is still a work in progress and further testing is needed.

Main improvements:

* Major refactor of compiler for performance and maintenance reasons
* Better fallback to object mode when native mode fails
* Improved worst case performance in object mode

The public interface of numba has been slightly changed. The idea is to
make it cleaner and more rational:

* jit decorator has been modified, so that it can be called without a signature.
  When called without a signature, it behaves as the old autojit. Autojit
  has been deprecated in favour of this approach.
* Jitted functions can now be overloaded.
* Added a "njit" decorator that behaves like "jit" decorator with nopython=True.
* The numba.vectorize namespace is gone. The vectorize decorator will
  be in the main numba namespace.
* Added a guvectorize decorator in the main numba namespace. It is
  similar to numba.vectorize, but takes a dimension signature. It
  generates gufuncs. This is a replacement for the GUVectorize gufunc
  factory which has been deprecated.

Main regressions (will be fixed in a future release):

* Creating new NumPy arrays is not supported in nopython mode
* Returning NumPy arrays is not supported in nopython mode
* NumPy array slicing is not supported in nopython mode
* lists and tuples are not supported in nopython mode
* string, datetime, cdecimal, and struct types are not implemented yet
* Extension types (classes) are not supported in nopython mode
* Closures are not supported
* Raise keyword is not supported
* Recursion is not support in nopython mode

Version 0.11
------------
* Experimental support for NumPy datetime type

Version 0.10
------------
* Annotation tool (./bin/numba --annotate --fancy) (thanks to Jay Bourque)
* Open sourced prange
* Support for raise statement
* Pluggable array representation
* Support for enumerate and zip (thanks to Eugene Toder)
* Better string formatting support (thanks to Eugene Toder)
* Builtins min(), max() and bool() (thanks to Eugene Toder)
* Fix some code reloading issues (thanks to Björn Linse)
* Recognize NumPy scalar objects (thanks to Björn Linse)


Version 0.9
-----------
* Improved math support
* Open sourced generalized ufuncs
* Improved array expressions

Version 0.8
-----------
* Support for autojit classes
    * Inheritance not yet supported
* Python 3 support for pycc
* Allow retrieval of ctypes function wrapper
    * And hence support retrieval of a pointer to the function
* Fixed a memory leak of array slicing views

Version 0.7.2
-------------
* Official Python 3 support (python 3.2 and 3.3)
* Support for intrinsics and instructions
* Various bug fixes (see https://github.com/numba/numba/issues?milestone=7&state=closed)

Version 0.7.1
-------------
* Various bug fixes

Version 0.7
-----------
* Open sourced single-threaded ufunc vectorizer
* Open sourced NumPy array expression compilation
* Open sourced fast NumPy array slicing
* Experimental Python 3 support
* Support for typed containers
    * typed lists and tuples
* Support for iteration over objects
* Support object comparisons
* Preliminary CFFI support
    * Jit calls to CFFI functions (passed into autojit functions)
    * TODO: Recognize ffi_lib.my_func attributes
* Improved support for ctypes
* Allow declaring extension attribute types as through class attributes
* Support for type casting in Python
    * Get the same semantics with or without numba compilation
* Support for recursion
    * For jit methods and extension classes
* Allow jit functions as C callbacks
* Friendlier error reporting
* Internal improvements
* A variety of bug fixes

Version 0.6.1
--------------
* Support for bitwise operations

Version 0.6
--------------
* Python 2.6 support
* Programmable typing
    * Allow users to add type inference for external code
* Better NumPy type inference
    * outer, inner, dot, vdot, tensordot, nonzero, where,
      binary ufuncs + methods (reduce, accumulate, reduceat, outer)
* Type based alias analysis
    * Support for strict aliasing
* Much faster autojit dispatch when calling from Python
* Faster numerical loops through data and stride pre-loading
* Integral overflow and underflow checking for conversions from objects
* Make Meta dependency optional

Version 0.5
--------------
* SSA-based type inference
    * Allows variable reuse
    * Allow referring to variables before lexical definition
* Support multiple comparisons
* Support for template types
* List comprehensions
* Support for pointers
* Many bug fixes
* Added user documentation

Version 0.4
--------------

Version 0.3.2
--------------

* Add support for object arithmetic (issue 56).
* Bug fixes (issue 55).

Version 0.3
--------------
* Changed default compilation approach to ast
* Added support for cross-module linking
* Added support for closures (can jit inner functions and return them) (see examples/closure.py)
* Added support for dtype structures (can access elements of structure with attribute access) (see examples/structures.py)
* Added support for extension types (numba classes) (see examples/numbaclasses.py)
* Added support for general Python code (use nopython to raise an error if Python C-API is used to avoid unexpected slowness because of lack of implementation defaulting to generic Python)
* Fixed many bugs
* Added support to detect math operations.
* Added with python and with nopython contexts
* Added more examples

Many features need to be documented still.  Look at examples and tests for more information.


Version 0.2
--------------
* Added an ast approach to compilation
* Removed d, f, i, b from numba namespace (use f8, f4, i4, b1)
* Changed function to autojit2
* Added autojit function to decorate calls to the function and use types of the variable to create compiled versions.
* changed keyword arguments to jit and autojit functions to restype and argtypes to be consistent with ctypes module.
* Added pycc -- a python to shared library compiler
