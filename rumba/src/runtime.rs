use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::artifact::CompiledArtifact;
use crate::errors::{compilation, unsupported};
use crate::types::ScalarType;

pub(crate) fn call_native(
    py: Python<'_>,
    artifact: &CompiledArtifact,
    args: &Bound<'_, PyTuple>,
) -> PyResult<PyObject> {
    match (artifact.return_type, artifact.signature.as_slice()) {
        (ScalarType::Int64, [ScalarType::Int64, ScalarType::Int64]) => unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(func(args.get_item(0)?.extract()?, args.get_item(1)?.extract()?).into_py(py))
        },
        (ScalarType::Float64, [ScalarType::Float64, ScalarType::Float64]) => unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(f64, f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(func(args.get_item(0)?.extract()?, args.get_item(1)?.extract()?).into_py(py))
        },
        (ScalarType::Int64, [ScalarType::Int64]) => unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(func(args.get_item(0)?.extract()?).into_py(py))
        },
        (ScalarType::Float64, [ScalarType::Float64]) => unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(func(args.get_item(0)?.extract()?).into_py(py))
        },
        (ScalarType::Bool, [ScalarType::Bool]) => unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(bool) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(func(args.get_item(0)?.extract()?).into_py(py))
        },
        (ScalarType::Int64, [ScalarType::Int64, ScalarType::Int64, ScalarType::Int64]) => unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(i64, i64, i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(func(
                args.get_item(0)?.extract()?,
                args.get_item(1)?.extract()?,
                args.get_item(2)?.extract()?,
            )
            .into_py(py))
        },
        _ => Err(unsupported(
            "native invocation currently supports homogeneous scalar signatures up to three arguments",
        )),
    }
}

fn load_error(err: libloading::Error) -> PyErr {
    compilation(format!("failed to load compiled symbol: {err}"))
}
