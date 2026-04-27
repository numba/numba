use std::path::PathBuf;
use std::sync::Arc;

use libloading::Library;
use pyo3::prelude::*;

use crate::types::ScalarType;

#[derive(Clone)]
pub(crate) struct CompiledArtifact {
    pub(crate) key: String,
    pub(crate) signature: Vec<ScalarType>,
    pub(crate) return_type: ScalarType,
    pub(crate) source: String,
    pub(crate) library_path: PathBuf,
    pub(crate) compile_command: Vec<String>,
    pub(crate) library: Arc<Library>,
}

#[pyclass(name = "CompiledArtifact")]
pub(crate) struct PyCompiledArtifact {
    pub(crate) inner: CompiledArtifact,
}

#[pymethods]
impl PyCompiledArtifact {
    #[getter]
    fn key(&self) -> String {
        self.inner.key.clone()
    }

    #[getter]
    fn library_path(&self) -> String {
        self.inner.library_path.display().to_string()
    }

    #[getter]
    fn compile_command(&self) -> Vec<String> {
        self.inner.compile_command.clone()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompiledArtifact>()?;
    Ok(())
}
