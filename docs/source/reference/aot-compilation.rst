Ahead-of-Time compilation
=========================

.. currentmodule:: numba.pycc

.. class:: CC(extension_name, source_module=None)

   An object used to generate compiled extensions from Numba-compiled
   Python functions.  *extension_name* is the name of the extension
   to be generated.  *source_module* is the Python module
   containing the functions; if ``None``, it is inferred by examining
   the call stack.

   :class:`CC` instances have the following attributes and methods:

   .. attribute:: name

      (read-only attribute) The name of the extension module to be generated.

   .. attribute:: output_dir

      (read-write attribute) The directory the extension module will be
      written into.  By default it is the directory the *source_module* is
      located in.

   .. attribute:: output_file

      (read-write attribute) The name of the file the extension module will
      be written to.  By default this follows the Python naming convention
      for the current platform.

   .. attribute:: target_cpu

      (read-write attribute) The name of the CPU model to generate code for.
      This will select the appropriate instruction set extensions.  By
      default, a generic CPU is selected in order to produce portable code.

      Recognized names for this attribute depend on the current architecture
      and LLVM version.  If you have LLVM installed, ``llc -mcpu=help``
      will give you a list.  Examples on x86-64 are ``"ivybridge"``,
      ``"haswell"``, ``"skylake"`` or ``"broadwell"``.  You can also give
      the value ``"host"`` which will select the current host CPU.

   .. attribute:: verbose

      (read-write attribute) If true, print out information while
      compiling the extension.  False by default.

   .. decorator:: export(exported_name, sig)

      Mark the decorated function for compilation with the signature *sig*.
      The compiled function will be exposed as *exported_name* in the
      generated extension module.

      All exported names within a given :class:`CC` instance must be
      distinct, otherwise an exception is raised.

   .. method:: compile()

      Compile all exported functions and generate the extension module
      as specified by :attr:`output_dir` and :attr:`output_file`.

   .. method:: distutils_extension(**kwargs)

      Return a :py:class:`distutils.core.Extension` instance allowing
      to integrate generation of the extension module in a conventional
      ``setup.py``-driven build process.  The optional *kwargs* let you
      pass optional parameters to the :py:class:`~distutils.core.Extension`
      constructor.

      In this mode of operation, it is not necessary to call :meth:`compile`
      yourself.  Also, :attr:`output_dir` and :attr:`output_file` will be
      ignored.

