#![allow(unexpected_cfgs)]

mod api;
mod artifact;
mod cache;
mod codegen;
mod compile;
mod dispatcher;
mod errors;
mod frontend;
mod ir;
mod runtime;
mod types;

use pyo3::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pymodule]
fn rumba(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    errors::register(py, m)?;
    artifact::register(m)?;
    dispatcher::register(m)?;
    types::register(m)?;
    api::register(m)?;
    Ok(())
}
