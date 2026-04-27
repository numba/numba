use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::artifact::{CompiledArtifact, PyCompiledArtifact};
use crate::compile::compile_parsed_function;
use crate::errors::unsupported;
use crate::frontend::bytecode::inspect_bytecode;
use crate::frontend::parse_function_input;
use crate::frontend::python_ast::build_rumba_ast;
use crate::runtime::call_native;
use crate::types::{parse_signature_tuple, signature_tuple, ScalarType};

#[pyclass]
pub(crate) struct Dispatcher {
    py_func: Py<PyAny>,
    cache: bool,
    debug: bool,
    explicit_signature: Option<Vec<ScalarType>>,
    compiled: Vec<CompiledArtifact>,
}

#[pymethods]
impl Dispatcher {
    #[getter]
    fn py_func(&self, py: Python<'_>) -> PyObject {
        self.py_func.clone_ref(py).into()
    }

    #[getter]
    fn cache(&self) -> bool {
        self.cache
    }

    #[getter]
    fn debug(&self) -> bool {
        self.debug
    }

    #[getter]
    fn signatures(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyList::empty_bound(py);
        for artifact in &self.compiled {
            out.append(signature_tuple(py, &artifact.signature)?)?;
        }
        Ok(out.into())
    }

    #[getter]
    fn _compiled(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new_bound(py);
        for artifact in &self.compiled {
            out.set_item(
                signature_tuple(py, &artifact.signature)?,
                Py::new(
                    py,
                    PyCompiledArtifact {
                        inner: artifact.clone(),
                    },
                )?,
            )?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        if kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
            return Err(unsupported("keyword arguments are not supported"));
        }
        let signature = match &self.explicit_signature {
            Some(signature) => signature.clone(),
            None => args
                .iter()
                .map(|arg| ScalarType::from_arg(&arg))
                .collect::<PyResult<Vec<_>>>()?,
        };
        if signature.len() != args.len() {
            return Err(unsupported(
                "argument count does not match explicit signature",
            ));
        }
        let artifact_index = self.compile_if_needed(py, &signature)?;
        let artifact = &self.compiled[artifact_index];
        call_native(py, artifact, args)
    }

    fn inspect_bytecode(&self, py: Python<'_>) -> PyResult<PyObject> {
        inspect_bytecode(py, self.py_func.bind(py))
    }

    fn inspect_rumba_ast(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ir = build_rumba_ast(py, self.py_func.bind(py))?;
        let out = PyDict::new_bound(py);
        out.set_item("name", ir.name)?;
        out.set_item("args", ir.args)?;
        let body = PyList::empty_bound(py);
        for stmt in ir.body.iter() {
            body.append(stmt.kind())?;
        }
        out.set_item("body", body)?;
        Ok(out.into())
    }

    #[pyo3(signature = (signature=None))]
    fn inspect_c(&self, signature: Option<&Bound<'_, PyTuple>>) -> PyResult<String> {
        if let Some(signature) = signature {
            let signature = parse_signature_tuple(signature)?;
            return self
                .compiled
                .iter()
                .find(|artifact| artifact.signature == signature)
                .map(|artifact| artifact.source.clone())
                .ok_or_else(|| unsupported("signature has not been compiled"));
        }
        if self.compiled.len() != 1 {
            return Err(unsupported(
                "inspect_c requires a signature before compilation",
            ));
        }
        Ok(self.compiled[0].source.clone())
    }
}

impl Dispatcher {
    pub(crate) fn new(
        py_func: Py<PyAny>,
        cache: bool,
        debug: bool,
        explicit_signature: Option<Vec<ScalarType>>,
    ) -> Self {
        Self {
            py_func,
            cache,
            debug,
            explicit_signature,
            compiled: Vec::new(),
        }
    }

    fn compile_if_needed(&mut self, py: Python<'_>, signature: &[ScalarType]) -> PyResult<usize> {
        if let Some(index) = self
            .compiled
            .iter()
            .position(|artifact| artifact.signature == signature)
        {
            return Ok(index);
        }

        let parsed = parse_function_input(py, self.py_func.bind(py))?;
        let artifact = compile_parsed_function(parsed, signature.to_vec())?;
        self.compiled.push(artifact);
        Ok(self.compiled.len() - 1)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dispatcher>()?;
    Ok(())
}
