use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyString, PyTuple};

use crate::errors::unsupported;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ScalarType {
    Int64,
    Float64,
    Bool,
}

impl ScalarType {
    pub(crate) fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "int64" => Ok(Self::Int64),
            "float64" => Ok(Self::Float64),
            "bool" => Ok(Self::Bool),
            other => Err(unsupported(format!("unsupported signature type {other:?}"))),
        }
    }

    pub(crate) fn from_arg(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        if arg.downcast::<PyBool>().is_ok() {
            Ok(Self::Bool)
        } else if arg.extract::<i64>().is_ok() {
            Ok(Self::Int64)
        } else if arg.extract::<f64>().is_ok() {
            Ok(Self::Float64)
        } else {
            Err(unsupported(format!(
                "unsupported argument type {}; supported scalar types are int, float, and bool",
                arg.get_type().name()?
            )))
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Int64 => "int64",
            Self::Float64 => "float64",
            Self::Bool => "bool",
        }
    }

    pub(crate) fn c_type(self) -> &'static str {
        match self {
            Self::Int64 => "int64_t",
            Self::Float64 => "double",
            Self::Bool => "bool",
        }
    }
}

#[pyclass(name = "ScalarType", frozen)]
#[derive(Clone)]
pub(crate) struct PyScalarType {
    pub(crate) typ: ScalarType,
}

#[pymethods]
impl PyScalarType {
    #[getter]
    fn name(&self) -> &'static str {
        self.typ.name()
    }

    fn __repr__(&self) -> String {
        format!("rumba.{}", self.typ.name())
    }

    fn __richcmp__(&self, other: PyRef<'_, PyScalarType>, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.typ == other.typ,
            CompareOp::Ne => self.typ != other.typ,
            _ => false,
        }
    }

    fn __hash__(&self) -> isize {
        match self.typ {
            ScalarType::Int64 => 1,
            ScalarType::Float64 => 2,
            ScalarType::Bool => 3,
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScalarType>()?;
    Ok(())
}

pub(crate) fn parse_signature_value(value: &Bound<'_, PyAny>) -> PyResult<Vec<ScalarType>> {
    if let Ok(value) = value.downcast::<PyString>() {
        let value = value.to_str()?;
        return value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ScalarType::from_name)
            .collect();
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        return tuple
            .iter()
            .map(|part| parse_signature_part(&part))
            .collect();
    }
    if let Ok(list) = value.downcast::<PyList>() {
        return list
            .iter()
            .map(|part| parse_signature_part(&part))
            .collect();
    }
    Err(unsupported("signature must be a string, tuple, or list"))
}

pub(crate) fn parse_signature_part(value: &Bound<'_, PyAny>) -> PyResult<ScalarType> {
    if let Ok(value) = value.extract::<PyRef<'_, PyScalarType>>() {
        return Ok(value.typ);
    }
    if let Ok(value) = value.downcast::<PyString>() {
        return ScalarType::from_name(value.to_str()?);
    }
    let repr = value.repr()?.extract::<String>()?;
    match repr.as_str() {
        "<class 'int'>" => Ok(ScalarType::Int64),
        "<class 'float'>" => Ok(ScalarType::Float64),
        "<class 'bool'>" => Ok(ScalarType::Bool),
        _ => Err(unsupported(format!("unsupported signature type {repr}"))),
    }
}

pub(crate) fn parse_signature_tuple(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<ScalarType>> {
    tuple
        .iter()
        .map(|item| parse_signature_part(&item))
        .collect()
}

pub(crate) fn signature_tuple(py: Python<'_>, signature: &[ScalarType]) -> PyResult<PyObject> {
    let items = signature
        .iter()
        .map(|typ| Py::new(py, PyScalarType { typ: *typ }).map(|obj| obj.into_py(py)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new_bound(py, items).into())
}

pub(crate) fn promote_numeric(
    left: ScalarType,
    right: ScalarType,
    op: impl AsRef<str>,
) -> ScalarType {
    match op.as_ref() {
        "Eq" | "NotEq" | "Lt" | "LtE" | "Gt" | "GtE" => ScalarType::Bool,
        "Div" => ScalarType::Float64,
        _ if left == ScalarType::Float64 || right == ScalarType::Float64 => ScalarType::Float64,
        _ if left == ScalarType::Bool && right == ScalarType::Bool => ScalarType::Bool,
        _ => ScalarType::Int64,
    }
}
