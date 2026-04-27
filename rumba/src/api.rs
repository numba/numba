use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dispatcher::Dispatcher;
use crate::errors::unsupported;
use crate::frontend::validate_function;
use crate::types::{parse_signature_value, ScalarType};
use crate::VERSION;

#[pyclass]
struct NjitDecorator {
    cache: bool,
    debug: bool,
    signature: Option<Vec<ScalarType>>,
}

#[pymethods]
impl NjitDecorator {
    fn __call__(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<Py<Dispatcher>> {
        validate_function(func.bind(py))?;
        Py::new(
            py,
            Dispatcher::new(func, self.cache, self.debug, self.signature.clone()),
        )
    }
}

#[pyfunction(signature = (fn_obj=None, **options))]
fn njit(
    py: Python<'_>,
    fn_obj: Option<Py<PyAny>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let options = parse_options(options)?;
    match fn_obj {
        Some(func) => {
            validate_function(func.bind(py))?;
            Ok(Py::new(
                py,
                Dispatcher::new(func, options.cache, options.debug, options.signature),
            )?
            .into_py(py))
        }
        None => Ok(Py::new(
            py,
            NjitDecorator {
                cache: options.cache,
                debug: options.debug,
                signature: options.signature,
            },
        )?
        .into_py(py)),
    }
}

#[pyfunction(signature = (fn_obj=None, **options))]
fn jit(
    py: Python<'_>,
    fn_obj: Option<Py<PyAny>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    njit(py, fn_obj, options)
}

#[pyfunction]
fn version() -> &'static str {
    VERSION
}

struct Options {
    cache: bool,
    debug: bool,
    signature: Option<Vec<ScalarType>>,
}

fn parse_options(options: Option<&Bound<'_, PyDict>>) -> PyResult<Options> {
    let mut parsed = Options {
        cache: false,
        debug: false,
        signature: None,
    };
    let Some(options) = options else {
        return Ok(parsed);
    };

    for key in options.keys().iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "cache" | "debug" | "signature" => {}
            _ => return Err(unsupported(format!("unsupported njit option(s): {key}"))),
        }
    }
    if let Some(value) = options.get_item("cache")? {
        parsed.cache = value.extract()?;
    }
    if let Some(value) = options.get_item("debug")? {
        parsed.debug = value.extract()?;
    }
    if let Some(value) = options.get_item("signature")? {
        parsed.signature = Some(parse_signature_value(&value)?);
    }
    Ok(parsed)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(njit, m)?)?;
    m.add_function(wrap_pyfunction!(jit, m)?)?;
    Ok(())
}
