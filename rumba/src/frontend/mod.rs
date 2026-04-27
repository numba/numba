pub(crate) mod bytecode;
pub(crate) mod python_ast;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::ir::ParsedFunction;

pub(crate) struct ParsedInput {
    pub(crate) function: ParsedFunction,
    pub(crate) metadata: CodeMetadata,
}

pub(crate) struct CodeMetadata {
    pub(crate) bytecode: Vec<u8>,
    pub(crate) consts: String,
    pub(crate) names: String,
    pub(crate) python: String,
    pub(crate) platform: String,
}

pub(crate) fn parse_function_input(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
) -> PyResult<ParsedInput> {
    Ok(ParsedInput {
        function: python_ast::build_rumba_ast(py, func)?,
        metadata: code_metadata(py, func)?,
    })
}

pub(crate) fn validate_function(func: &Bound<'_, PyAny>) -> PyResult<()> {
    let inspect = func.py().import_bound("inspect")?;
    if inspect.call_method1("isfunction", (func,))?.extract()? {
        Ok(())
    } else {
        Err(crate::errors::unsupported(
            "@rumba.njit can only decorate Python functions",
        ))
    }
}

fn code_metadata(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<CodeMetadata> {
    let code = func.getattr("__code__")?;
    let bytecode = code
        .getattr("co_code")?
        .downcast_into::<PyBytes>()?
        .as_bytes()
        .to_vec();
    Ok(CodeMetadata {
        bytecode,
        consts: code.getattr("co_consts")?.repr()?.extract()?,
        names: code.getattr("co_names")?.repr()?.extract()?,
        python: py
            .import_bound("sys")?
            .getattr("version_info")?
            .repr()?
            .extract()?,
        platform: py
            .import_bound("platform")?
            .call_method0("platform")?
            .extract()?,
    })
}

pub(crate) fn getattr<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr(name)
}

pub(crate) fn class_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type().name().map(|name| name.to_string())
}
