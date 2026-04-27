use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(rumba, RumbaError, PyException);
create_exception!(rumba, RumbaUnsupportedError, RumbaError);
create_exception!(rumba, RumbaCompilationError, RumbaError);

pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RumbaError", py.get_type_bound::<RumbaError>())?;
    m.add(
        "RumbaUnsupportedError",
        py.get_type_bound::<RumbaUnsupportedError>(),
    )?;
    m.add(
        "RumbaCompilationError",
        py.get_type_bound::<RumbaCompilationError>(),
    )?;
    Ok(())
}

pub(crate) fn unsupported(message: impl Into<String>) -> PyErr {
    RumbaUnsupportedError::new_err(message.into())
}

pub(crate) fn compilation(message: impl Into<String>) -> PyErr {
    RumbaCompilationError::new_err(message.into())
}
