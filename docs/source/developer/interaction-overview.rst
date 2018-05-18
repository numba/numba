
Overview of Python-Numba interaction
====================================

This is a conceptual representation of the relationship of different elements in Python and Numba.
This is not meant to be an explicit class diagram.

.. uml::
  @startuml
  namespace Python {
    namespace built_in {
      class PyObject {
        ref_count
      }

      class Array
      class List
      class Buffer

      PyObject +-- Array
      PyObject +-- List
      PyObject +-- Buffer
    }
  }

  namespace Numba {
    class Builder
    class Context

    namespace Models {
      class PrimitiveModel
      class StructModel

    }

    namespace LLVMlite {

      class LiteralStructType
      class ArrayType
      class IntType
      class DoubleType
      class FloatType
      class Constant
      class PointerType
      class FunctionType
    }

  }

  Numba.LLVMlite.IntType <-- Numba.Models.PrimitiveModel
  Numba.LLVMlite.DoubleType <-- Numba.Models.PrimitiveModel
  Numba.LLVMlite.FloatType <-- Numba.Models.PrimitiveModel
  Numba.LLVMlite.IntType <-- Numba.Models.PrimitiveModel
  @enduml