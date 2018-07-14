"""
Implements a Python representation of ndt function types.

The intent is to be able to:
    1. Parse an existing NDT function signature info a `Function` object
    2. Optionally transform that object, by filling in it's return dtype
    3. Convert back into an ndt function signature and a numba function signature.
"""

from typing import NamedTuple, Tuple, Union
from .. import types as numba_types

class Ellipsis_(NamedTuple):
    as_ndt = '...'


class NamedEllipsis(NamedTuple):
    name: str

    @property
    def as_ndt(self):
        return f'{self.name}...'

    @classmethod
    def from_ndt(cls, ndt: str):
        return cls(ndt.split('...')[0])

class VarEllipsis(NamedTuple):
    as_ndt = 'var...'


class DType(NamedTuple):
    dtype: str

    @property
    def as_ndt(self):
        return self.dtype

    @property
    def as_numba(self):
        return getattr(numba_types, self.dtype)

class DTypeVariable(NamedTuple):
    name: str

    @property
    def as_ndt(self):
        return self.name

class SymbolicDimension(NamedTuple):
    name: str

    @property
    def as_ndt(self):
        return self.name

class FixedDimension(NamedTuple):
    length: int

    @property
    def as_ndt(self):
        return str(self.length)

    @classmethod
    def from_ndt(cls, ndt):
        return cls(int(ndt))

class Array(NamedTuple):
    ellipsis: Union[None, NamedEllipsis, VarEllipsis, Ellipsis_]
    dims: Tuple[Union[SymbolicDimension, FixedDimension], ...]
    dtype: Union[DType, DTypeVariable]

    @property
    def as_ndt(self):
        all_dims = list(self.dims) + [self.dtype]
        if self.ellipsis is not None:
            all_dims = [self.ellipsis] + all_dims
        return ' * '.join(i.as_ndt for i in all_dims)

    @property
    def ndim(self):
        return len(self.dims)

    @classmethod
    def from_ndt(cls, ndt: str):
        *dims_str, dtype_str = ndt.split(' * ')

        if len(dims_str) > 0:
            possible_ellipsis, *dims_str = dims_str
            if '...' == possible_ellipsis:
                ellipsis = Ellipsis_()
            elif 'var...' == possible_ellipsis:
                ellipsis = VarEllipsis()
            elif '...' in possible_ellipsis:
                ellipsis = NamedEllipsis.from_ndt(possible_ellipsis)
            else:
                ellipsis = None
                dims_str = [possible_ellipsis] + dims_str
        else:
            ellipsis = None

        dims = []
        for dim_str in dims_str:
            try:
                dim = FixedDimension.from_ndt(dim_str)
            except ValueError:
                dim = SymbolicDimension(dim_str)
            dims.append(dim)

        if dtype_str.islower():
            dtype = DType(dtype_str)
        else:
            dtype = DTypeVariable(dtype_str)

        return cls(ellipsis=ellipsis, dims=tuple(dims), dtype=dtype)

    @property
    def as_numba(self):
        if self.ndim == 0:
            return self.dtype.as_numba
        return numba_types.npytypes.Array(self.dtype.as_numba, self.ndim, 'A')

    @property
    def is_concrete(self):
        return isinstance(self.dtype, DType)

class FunctionInputs(NamedTuple):
    values: Tuple[Array, ...]

    @property
    def as_ndt(self):
        return ', '.join(v.as_ndt for v in self.values)

    @classmethod
    def from_ndt(cls, ndt: str):
        return cls(
            values=tuple(map(Array.from_ndt, ndt.split(', '))),
        )

    @property
    def as_numba(self):
        return tuple(v.as_numba for v in self.values)


class Function(NamedTuple):
    inputs: FunctionInputs
    output: Array

    @property
    def returns_scalar(self) -> bool:
        return self.output.ndim == 0

    @property
    def as_ndt(self) -> str:
        return f"{self.inputs.as_ndt} -> {self.output.as_ndt}"

    @classmethod
    def from_ndt(cls, ndt: str):
        inputs, output = ndt.split(' -> ')
        return cls(
            inputs=FunctionInputs.from_ndt(inputs),
            output=Array.from_ndt(output)
        )

    @classmethod
    def zero_dim(cls, dtypes):
        return cls(
            inputs=FunctionInputs(tuple(Array(Ellipsis_, tuple(), d) for d in dtypes)),
            output=Array(Ellipsis_, tuple(), DTypeVariable('D'))
        )

    @property
    def as_numba(self) -> str:
        if self.returns_scalar:
            if self.output.is_concrete:
                return self.output.as_numba(*self.inputs.as_numba)
            return self.inputs.as_numba
        return (*self.inputs.as_numba, self.output.as_numba)

    @property
    def dimensions(self) -> int:
        return [v.ndim for v in self.inputs.values] + [self.output.ndim]

    def make_output_concrete(self, output_dtype):
        if isinstance(self.output.dtype, DTypeVariable):
            return self._replace(output=self.output._replace(dtype=output_dtype))
        if self.output.dtype != output_dtype:
            raise Exception(
                f"Attempting to replace output's existing dtype of {self.output.dtype} with {output_dtype}"
            )
        return self
