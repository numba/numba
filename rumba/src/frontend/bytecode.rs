use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) fn inspect_bytecode(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let dis = py.import_bound("dis")?;
    let bytecode = dis.call_method1("Bytecode", (func,))?;
    let out = PyList::empty_bound(py);
    for inst in bytecode.iter()? {
        let inst = inst?;
        let item = PyDict::new_bound(py);
        item.set_item("offset", inst.getattr("offset")?)?;
        item.set_item("opname", inst.getattr("opname")?)?;
        item.set_item("argrepr", inst.getattr("argrepr")?)?;
        item.set_item("starts_line", inst.getattr("starts_line")?)?;
        out.append(item)?;
    }
    Ok(out.into())
}
