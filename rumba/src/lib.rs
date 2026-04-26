#![allow(unexpected_cfgs)]

use libloading::Library;
use pyo3::basic::CompareOp;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

create_exception!(rumba, RumbaError, PyException);
create_exception!(rumba, RumbaUnsupportedError, RumbaError);
create_exception!(rumba, RumbaCompilationError, RumbaError);

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ScalarType {
    Int64,
    Float64,
    Bool,
}

impl ScalarType {
    fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "int64" => Ok(Self::Int64),
            "float64" => Ok(Self::Float64),
            "bool" => Ok(Self::Bool),
            other => Err(unsupported(format!("unsupported signature type {other:?}"))),
        }
    }

    fn from_arg(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
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

    fn name(self) -> &'static str {
        match self {
            Self::Int64 => "int64",
            Self::Float64 => "float64",
            Self::Bool => "bool",
        }
    }

    fn c_type(self) -> &'static str {
        match self {
            Self::Int64 => "int64_t",
            Self::Float64 => "double",
            Self::Bool => "bool",
        }
    }
}

#[pyclass(name = "ScalarType", frozen)]
#[derive(Clone)]
struct PyScalarType {
    typ: ScalarType,
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

struct CExpr {
    code: String,
    typ: ScalarType,
}

struct ParsedFunction {
    name: String,
    args: Vec<String>,
    body: Vec<StmtNode>,
}

enum StmtNode {
    Return(ExprNode),
    Assign {
        name: String,
        value: ExprNode,
    },
    AugAssign {
        name: String,
        op: BinOp,
        value: ExprNode,
    },
    If {
        test: ExprNode,
        body: Vec<StmtNode>,
        orelse: Vec<StmtNode>,
    },
    ForRange {
        target: String,
        start: ExprNode,
        stop: ExprNode,
        step: ExprNode,
        body: Vec<StmtNode>,
    },
}

impl StmtNode {
    fn kind(&self) -> &'static str {
        match self {
            Self::Return(_) => "Return",
            Self::Assign { .. } => "Assign",
            Self::AugAssign { .. } => "AugAssign",
            Self::If { .. } => "If",
            Self::ForRange { .. } => "For",
        }
    }
}

enum ExprNode {
    Constant(ConstantValue),
    Name(String),
    BinOp {
        left: Box<ExprNode>,
        op: BinOp,
        right: Box<ExprNode>,
    },
    UnaryOp {
        op: UnaryOp,
        value: Box<ExprNode>,
    },
    Compare {
        left: Box<ExprNode>,
        op: CmpOp,
        right: Box<ExprNode>,
    },
}

enum ConstantValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
}

impl BinOp {
    fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mult => "*",
            Self::Div => "/",
            Self::FloorDiv => "/",
            Self::Mod => "%",
        }
    }

    fn type_name(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mult => "Mult",
            Self::Div => "Div",
            Self::FloorDiv => "FloorDiv",
            Self::Mod => "Mod",
        }
    }
}

enum UnaryOp {
    Not,
    USub,
    UAdd,
}

enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
}

impl CmpOp {
    fn symbol(&self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtE => "<=",
            Self::Gt => ">",
            Self::GtE => ">=",
        }
    }
}

struct Emitter {
    function_ir: ParsedFunction,
    signature: Vec<ScalarType>,
    env: HashMap<String, ScalarType>,
    declared: HashSet<String>,
    lines: Vec<String>,
    return_type: Option<ScalarType>,
}

impl Emitter {
    fn new(function_ir: ParsedFunction, signature: Vec<ScalarType>) -> PyResult<Self> {
        if function_ir.args.len() != signature.len() {
            return Err(unsupported("argument count does not match signature"));
        }

        let env = function_ir
            .args
            .iter()
            .cloned()
            .zip(signature.iter().copied())
            .collect::<HashMap<_, _>>();
        let declared = function_ir.args.iter().cloned().collect::<HashSet<_>>();

        Ok(Self {
            function_ir,
            signature,
            env,
            declared,
            lines: Vec::new(),
            return_type: None,
        })
    }

    fn emit(&mut self) -> PyResult<(String, ScalarType)> {
        let params = self
            .function_ir
            .args
            .iter()
            .zip(self.signature.iter())
            .map(|(name, typ)| format!("{} {}", typ.c_type(), name))
            .collect::<Vec<_>>()
            .join(", ");

        self.lines.push("#include <stdbool.h>".to_string());
        self.lines.push("#include <stdint.h>".to_string());
        self.lines.push(String::new());
        self.lines.push(format!(
            "__attribute__((visibility(\"default\"))) int64_t rumba_entry({params}) {{"
        ));

        let body = std::mem::take(&mut self.function_ir.body);
        for stmt in &body {
            self.stmt(stmt, 1)?;
        }
        self.function_ir.body = body;

        let return_type = self
            .return_type
            .ok_or_else(|| unsupported("function must return a scalar value"))?;
        self.lines[3] = format!(
            "__attribute__((visibility(\"default\"))) {} rumba_entry({params}) {{",
            return_type.c_type()
        );
        self.lines.push("}".to_string());

        Ok((format!("{}\n", self.lines.join("\n")), return_type))
    }

    fn stmt(&mut self, node: &StmtNode, level: usize) -> PyResult<()> {
        match node {
            StmtNode::Return(value) => {
                let expr = self.expr(value)?;
                self.return_type = Some(match self.return_type {
                    None => expr.typ,
                    Some(current) if current == expr.typ => current,
                    Some(current) => promote_numeric(current, expr.typ, "Add"),
                });
                self.lines
                    .push(format!("{}return {};", indent(level), expr.code));
                Ok(())
            }
            StmtNode::Assign { name, value } => {
                let expr = self.expr(value)?;
                self.env.insert(name.clone(), expr.typ);
                let prefix = if self.declared.contains(name) {
                    String::new()
                } else {
                    self.declared.insert(name.clone());
                    format!("{} ", expr.typ.c_type())
                };
                self.lines
                    .push(format!("{}{prefix}{name} = {};", indent(level), expr.code));
                Ok(())
            }
            StmtNode::AugAssign { name, op, value } => {
                if !self.env.contains_key(name) {
                    return Err(unsupported(format!(
                        "local variable {name:?} is used before assignment"
                    )));
                }
                let expr = self.expr(value)?;
                let op = op.symbol();
                self.lines
                    .push(format!("{}{name} {op}= {};", indent(level), expr.code));
                Ok(())
            }
            StmtNode::If { test, body, orelse } => {
                let test = self.expr(test)?;
                if test.typ != ScalarType::Bool {
                    return Err(unsupported("if condition must be boolean"));
                }
                self.lines
                    .push(format!("{}if ({}) {{", indent(level), test.code));
                for child in body {
                    self.stmt(child, level + 1)?;
                }
                if !orelse.is_empty() {
                    self.lines.push(format!("{}}} else {{", indent(level)));
                    for child in orelse {
                        self.stmt(child, level + 1)?;
                    }
                }
                self.lines.push(format!("{}}}", indent(level)));
                Ok(())
            }
            StmtNode::ForRange {
                target,
                start,
                stop,
                step,
                body,
            } => self.for_range(target, start, stop, step, body, level),
        }
    }

    fn for_range(
        &mut self,
        name: &str,
        start: &ExprNode,
        stop: &ExprNode,
        step: &ExprNode,
        body: &[StmtNode],
        level: usize,
    ) -> PyResult<()> {
        let start = self.expr(start)?.code;
        let stop = self.expr(stop)?.code;
        let step = self.expr(step)?.code;
        self.env.insert(name.to_string(), ScalarType::Int64);
        let decl = if self.declared.contains(name) {
            String::new()
        } else {
            self.declared.insert(name.to_string());
            "int64_t ".to_string()
        };
        let cmp = if step.starts_with('-') { ">" } else { "<" };
        self.lines.push(format!(
            "{}for ({decl}{name} = {start}; {name} {cmp} {stop}; {name} += {step}) {{",
            indent(level)
        ));
        for child in body {
            self.stmt(child, level + 1)?;
        }
        self.lines.push(format!("{}}}", indent(level)));
        Ok(())
    }

    fn expr(&mut self, node: &ExprNode) -> PyResult<CExpr> {
        match node {
            ExprNode::Constant(value) => match value {
                ConstantValue::Bool(value) => Ok(CExpr {
                    code: if *value { "true" } else { "false" }.to_string(),
                    typ: ScalarType::Bool,
                }),
                ConstantValue::Int(value) => Ok(CExpr {
                    code: value.to_string(),
                    typ: ScalarType::Int64,
                }),
                ConstantValue::Float(value) => Ok(CExpr {
                    code: value.to_string(),
                    typ: ScalarType::Float64,
                }),
            },
            ExprNode::Name(name) => {
                let typ = self
                    .env
                    .get(name)
                    .copied()
                    .ok_or_else(|| unsupported(format!("unknown name {name:?}")))?;
                Ok(CExpr {
                    code: name.clone(),
                    typ,
                })
            }
            ExprNode::BinOp { left, op, right } => {
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                Ok(CExpr {
                    code: format!("({} {} {})", left.code, op.symbol(), right.code),
                    typ: promote_numeric(left.typ, right.typ, op.type_name()),
                })
            }
            ExprNode::UnaryOp { op, value } => {
                let operand = self.expr(value)?;
                match op {
                    UnaryOp::Not => Ok(CExpr {
                        code: format!("(!{})", operand.code),
                        typ: ScalarType::Bool,
                    }),
                    UnaryOp::USub => Ok(CExpr {
                        code: format!("(-{})", operand.code),
                        typ: operand.typ,
                    }),
                    UnaryOp::UAdd => Ok(CExpr {
                        code: format!("(+{})", operand.code),
                        typ: operand.typ,
                    }),
                }
            }
            ExprNode::Compare { left, op, right } => {
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                Ok(CExpr {
                    code: format!("({} {} {})", left.code, op.symbol(), right.code),
                    typ: ScalarType::Bool,
                })
            }
        }
    }
}

#[derive(Clone)]
struct CompiledArtifact {
    key: String,
    signature: Vec<ScalarType>,
    return_type: ScalarType,
    source: String,
    library_path: PathBuf,
    compile_command: Vec<String>,
    library: Arc<Library>,
}

#[pyclass(name = "CompiledArtifact")]
struct PyCompiledArtifact {
    inner: CompiledArtifact,
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

#[pyclass]
struct Dispatcher {
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
    fn new(
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

        let artifact = compile_function(py, self.py_func.bind(py), signature.to_vec())?;
        self.compiled.push(artifact);
        Ok(self.compiled.len() - 1)
    }
}

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

#[pymodule]
fn rumba(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add("RumbaError", py.get_type_bound::<RumbaError>())?;
    m.add(
        "RumbaUnsupportedError",
        py.get_type_bound::<RumbaUnsupportedError>(),
    )?;
    m.add(
        "RumbaCompilationError",
        py.get_type_bound::<RumbaCompilationError>(),
    )?;
    m.add_class::<Dispatcher>()?;
    m.add_class::<PyScalarType>()?;
    m.add_class::<PyCompiledArtifact>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(njit, m)?)?;
    m.add_function(wrap_pyfunction!(jit, m)?)?;
    Ok(())
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

fn parse_signature_value(value: &Bound<'_, PyAny>) -> PyResult<Vec<ScalarType>> {
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

fn parse_signature_part(value: &Bound<'_, PyAny>) -> PyResult<ScalarType> {
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

fn parse_signature_tuple(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<ScalarType>> {
    tuple
        .iter()
        .map(|item| parse_signature_part(&item))
        .collect()
}

fn validate_function(func: &Bound<'_, PyAny>) -> PyResult<()> {
    let inspect = func.py().import_bound("inspect")?;
    if inspect.call_method1("isfunction", (func,))?.extract()? {
        Ok(())
    } else {
        Err(unsupported(
            "@rumba.njit can only decorate Python functions",
        ))
    }
}

fn build_rumba_ast(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<ParsedFunction> {
    let code = func.getattr("__code__")?;
    let freevars = code.getattr("co_freevars")?.downcast_into::<PyTuple>()?;
    if !freevars.is_empty() {
        return Err(unsupported("closures are not supported"));
    }
    let kwonlyargcount: usize = code.getattr("co_kwonlyargcount")?.extract()?;
    if kwonlyargcount != 0 {
        return Err(unsupported("keyword-only arguments are not supported"));
    }
    let posonlyargcount: usize = code.getattr("co_posonlyargcount")?.extract()?;
    if posonlyargcount != 0 {
        return Err(unsupported("positional-only arguments are not supported"));
    }

    let inspect = py.import_bound("inspect")?;
    let textwrap = py.import_bound("textwrap")?;
    let ast = py.import_bound("ast")?;
    let source = inspect
        .call_method1("getsource", (func,))
        .map_err(|_| unsupported("source is unavailable for this function"))?;
    let source = textwrap.call_method1("dedent", (source,))?;
    let module = ast.call_method1("parse", (source,))?;
    let body = module.getattr("body")?.downcast_into::<PyList>()?;

    for node in body.iter() {
        if class_name(&node)? == "FunctionDef" {
            let args_node = node.getattr("args")?;
            if !args_node.getattr("vararg")?.is_none()
                || !args_node.getattr("kwarg")?.is_none()
                || !args_node
                    .getattr("defaults")?
                    .downcast_into::<PyList>()?
                    .is_empty()
            {
                return Err(unsupported(
                    "varargs, kwargs, and default arguments are not supported",
                ));
            }
            let args = args_node
                .getattr("args")?
                .downcast_into::<PyList>()?
                .iter()
                .map(|arg| arg.getattr("arg")?.extract::<String>())
                .collect::<PyResult<Vec<_>>>()?;
            let parsed_body = parse_stmt_list(&node.getattr("body")?.downcast_into::<PyList>()?)?;
            return Ok(ParsedFunction {
                name: node.getattr("name")?.extract()?,
                args,
                body: parsed_body,
            });
        }
    }
    Err(unsupported("expected a Python function"))
}

fn parse_stmt_list(body: &Bound<'_, PyList>) -> PyResult<Vec<StmtNode>> {
    body.iter().map(|stmt| parse_stmt(&stmt)).collect()
}

fn parse_stmt(node: &Bound<'_, PyAny>) -> PyResult<StmtNode> {
    match class_name(node)?.as_str() {
        "Return" => {
            let value = getattr(node, "value")?;
            if value.is_none() {
                return Err(unsupported("empty return is not supported"));
            }
            Ok(StmtNode::Return(parse_expr(&value)?))
        }
        "Assign" => {
            let targets = getattr(node, "targets")?.downcast_into::<PyList>()?;
            if targets.len() != 1 {
                return Err(unsupported("only simple local assignments are supported"));
            }
            let target = targets.get_item(0)?;
            if class_name(&target)? != "Name" {
                return Err(unsupported("only simple local assignments are supported"));
            }
            Ok(StmtNode::Assign {
                name: getattr(&target, "id")?.extract()?,
                value: parse_expr(&getattr(node, "value")?)?,
            })
        }
        "AugAssign" => {
            let target = getattr(node, "target")?;
            if class_name(&target)? != "Name" {
                return Err(unsupported(
                    "only augmented assignment to local names is supported",
                ));
            }
            Ok(StmtNode::AugAssign {
                name: getattr(&target, "id")?.extract()?,
                op: parse_binop(&getattr(node, "op")?)?,
                value: parse_expr(&getattr(node, "value")?)?,
            })
        }
        "If" => Ok(StmtNode::If {
            test: parse_expr(&getattr(node, "test")?)?,
            body: parse_stmt_list(&getattr(node, "body")?.downcast_into::<PyList>()?)?,
            orelse: parse_stmt_list(&getattr(node, "orelse")?.downcast_into::<PyList>()?)?,
        }),
        "For" => parse_for_range(node),
        other => Err(unsupported(format!("unsupported statement {other}"))),
    }
}

fn parse_for_range(node: &Bound<'_, PyAny>) -> PyResult<StmtNode> {
    let target = getattr(node, "target")?;
    if class_name(&target)? != "Name" {
        return Err(unsupported(
            "only simple range loop variables are supported",
        ));
    }

    let iter = getattr(node, "iter")?;
    if class_name(&iter)? != "Call" {
        return Err(unsupported("only range(...) loops are supported"));
    }
    let func = getattr(&iter, "func")?;
    if class_name(&func)? != "Name" {
        return Err(unsupported("only range(...) loops are supported"));
    }
    let func_name: String = getattr(&func, "id")?.extract()?;
    if func_name != "range" {
        return Err(unsupported("only range(...) loops are supported"));
    }

    let args = getattr(&iter, "args")?.downcast_into::<PyList>()?;
    let (start, stop, step) = match args.len() {
        1 => (
            ExprNode::Constant(ConstantValue::Int(0)),
            parse_expr(&args.get_item(0)?)?,
            ExprNode::Constant(ConstantValue::Int(1)),
        ),
        2 => (
            parse_expr(&args.get_item(0)?)?,
            parse_expr(&args.get_item(1)?)?,
            ExprNode::Constant(ConstantValue::Int(1)),
        ),
        3 => (
            parse_expr(&args.get_item(0)?)?,
            parse_expr(&args.get_item(1)?)?,
            parse_expr(&args.get_item(2)?)?,
        ),
        _ => return Err(unsupported("range accepts one to three arguments")),
    };

    Ok(StmtNode::ForRange {
        target: getattr(&target, "id")?.extract()?,
        start,
        stop,
        step,
        body: parse_stmt_list(&getattr(node, "body")?.downcast_into::<PyList>()?)?,
    })
}

fn parse_expr(node: &Bound<'_, PyAny>) -> PyResult<ExprNode> {
    match class_name(node)?.as_str() {
        "Constant" => {
            let value = getattr(node, "value")?;
            if let Ok(value) = value.extract::<bool>() {
                return Ok(ExprNode::Constant(ConstantValue::Bool(value)));
            }
            if let Ok(value) = value.extract::<i64>() {
                return Ok(ExprNode::Constant(ConstantValue::Int(value)));
            }
            if let Ok(value) = value.extract::<f64>() {
                return Ok(ExprNode::Constant(ConstantValue::Float(value)));
            }
            Err(unsupported("unsupported constant"))
        }
        "Name" => Ok(ExprNode::Name(getattr(node, "id")?.extract()?)),
        "BinOp" => Ok(ExprNode::BinOp {
            left: Box::new(parse_expr(&getattr(node, "left")?)?),
            op: parse_binop(&getattr(node, "op")?)?,
            right: Box::new(parse_expr(&getattr(node, "right")?)?),
        }),
        "UnaryOp" => Ok(ExprNode::UnaryOp {
            op: parse_unary_op(&getattr(node, "op")?)?,
            value: Box::new(parse_expr(&getattr(node, "operand")?)?),
        }),
        "Compare" => {
            let ops = getattr(node, "ops")?.downcast_into::<PyList>()?;
            let comparators = getattr(node, "comparators")?.downcast_into::<PyList>()?;
            if ops.len() != 1 || comparators.len() != 1 {
                return Err(unsupported("chained comparisons are not supported"));
            }
            Ok(ExprNode::Compare {
                left: Box::new(parse_expr(&getattr(node, "left")?)?),
                op: parse_cmpop(&ops.get_item(0)?)?,
                right: Box::new(parse_expr(&comparators.get_item(0)?)?),
            })
        }
        other => Err(unsupported(format!("unsupported expression {other}"))),
    }
}

fn parse_binop(op: &Bound<'_, PyAny>) -> PyResult<BinOp> {
    match class_name(op)?.as_str() {
        "Add" => Ok(BinOp::Add),
        "Sub" => Ok(BinOp::Sub),
        "Mult" => Ok(BinOp::Mult),
        "Div" => Ok(BinOp::Div),
        "FloorDiv" => Ok(BinOp::FloorDiv),
        "Mod" => Ok(BinOp::Mod),
        other => Err(unsupported(format!("unsupported binary operator {other}"))),
    }
}

fn parse_unary_op(op: &Bound<'_, PyAny>) -> PyResult<UnaryOp> {
    match class_name(op)?.as_str() {
        "Not" => Ok(UnaryOp::Not),
        "USub" => Ok(UnaryOp::USub),
        "UAdd" => Ok(UnaryOp::UAdd),
        other => Err(unsupported(format!("unsupported unary operator {other}"))),
    }
}

fn parse_cmpop(op: &Bound<'_, PyAny>) -> PyResult<CmpOp> {
    match class_name(op)?.as_str() {
        "Eq" => Ok(CmpOp::Eq),
        "NotEq" => Ok(CmpOp::NotEq),
        "Lt" => Ok(CmpOp::Lt),
        "LtE" => Ok(CmpOp::LtE),
        "Gt" => Ok(CmpOp::Gt),
        "GtE" => Ok(CmpOp::GtE),
        other => Err(unsupported(format!(
            "unsupported comparison operator {other}"
        ))),
    }
}

fn inspect_bytecode(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<PyObject> {
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

fn compile_function(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    signature: Vec<ScalarType>,
) -> PyResult<CompiledArtifact> {
    let function_ir = build_rumba_ast(py, func)?;
    let mut emitter = Emitter::new(function_ir, signature.clone())?;
    let (source, return_type) = emitter.emit()?;
    let metadata = code_metadata(py, func)?;
    let key = cache_key(&metadata, &signature, &source);
    let cache_dir = env::var("RUMBA_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("rumba-cache"));
    let build_dir = cache_dir.join(&key);
    fs::create_dir_all(&build_dir)
        .map_err(|err| compilation(format!("failed to create cache directory: {err}")))?;
    let source_path = build_dir.join("module.c");
    let library_path = build_dir.join(format!("module{}", shared_suffix()?));
    fs::write(&source_path, &source)
        .map_err(|err| compilation(format!("failed to write generated C source: {err}")))?;

    let cc = env::var("CC")
        .ok()
        .or_else(|| find_executable("cc"))
        .or_else(|| find_executable("clang"))
        .or_else(|| find_executable("gcc"))
        .ok_or_else(|| compilation("no C compiler found; set CC to a working compiler"))?;

    let command = vec![
        cc,
        "-shared".to_string(),
        "-fPIC".to_string(),
        "-O2".to_string(),
        source_path.display().to_string(),
        "-o".to_string(),
        library_path.display().to_string(),
    ];

    if !library_path.exists() {
        let output = Command::new(&command[0])
            .args(&command[1..])
            .output()
            .map_err(|err| compilation(format!("failed to run C compiler: {err}")))?;
        if !output.status.success() {
            return Err(compilation(format!(
                "C compilation failed:\ncommand: {}\nstdout: {}\nstderr: {}",
                command.join(" "),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )));
        }
    }

    let library = unsafe { Library::new(&library_path) }
        .map_err(|err| compilation(format!("failed to load shared library: {err}")))?;

    Ok(CompiledArtifact {
        key,
        signature,
        return_type,
        source,
        library_path,
        compile_command: command,
        library: Arc::new(library),
    })
}

fn call_native(
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

struct CodeMetadata {
    bytecode: Vec<u8>,
    consts: String,
    names: String,
    python: String,
    platform: String,
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

fn signature_tuple(py: Python<'_>, signature: &[ScalarType]) -> PyResult<PyObject> {
    let items = signature
        .iter()
        .map(|typ| Py::new(py, PyScalarType { typ: *typ }).map(|obj| obj.into_py(py)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new_bound(py, items).into())
}

fn getattr<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr(name)
}

fn class_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type().name().map(|name| name.to_string())
}

fn promote_numeric(left: ScalarType, right: ScalarType, op: impl AsRef<str>) -> ScalarType {
    match op.as_ref() {
        "Eq" | "NotEq" | "Lt" | "LtE" | "Gt" | "GtE" => ScalarType::Bool,
        "Div" => ScalarType::Float64,
        _ if left == ScalarType::Float64 || right == ScalarType::Float64 => ScalarType::Float64,
        _ if left == ScalarType::Bool && right == ScalarType::Bool => ScalarType::Bool,
        _ => ScalarType::Int64,
    }
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn cache_key(metadata: &CodeMetadata, signature: &[ScalarType], source: &str) -> String {
    let mut payload = String::new();
    write!(&mut payload, "bytecode={:x?};", metadata.bytecode)
        .expect("write to String cannot fail");
    write!(
        &mut payload,
        "consts={};names={};python={};platform={};rumba={};",
        metadata.consts, metadata.names, metadata.python, metadata.platform, VERSION
    )
    .expect("write to String cannot fail");
    payload.push_str("signature=");
    for typ in signature {
        payload.push_str(typ.name());
        payload.push(',');
    }
    payload.push_str(";source=");
    payload.push_str(source);
    stable_hex_hash(payload.as_bytes())
}

fn shared_suffix() -> PyResult<&'static str> {
    if cfg!(target_os = "macos") {
        Ok(".dylib")
    } else if cfg!(target_os = "linux") {
        Ok(".so")
    } else {
        Err(unsupported(
            "native compilation currently supports Linux and macOS",
        ))
    }
}

fn find_executable(name: &str) -> Option<String> {
    let paths = env::var_os("PATH")?;
    env::split_paths(&paths)
        .map(|path| path.join(name))
        .find(|candidate| candidate.is_file())
        .map(|candidate| candidate.display().to_string())
}

fn stable_hex_hash(bytes: &[u8]) -> String {
    let mut first = 0xcbf29ce484222325_u64;
    let mut second = 0x84222325cbf29ce4_u64;

    for byte in bytes {
        first ^= u64::from(*byte);
        first = first.wrapping_mul(0x100000001b3);

        second ^= u64::from(byte.rotate_left(1));
        second = second.wrapping_mul(0x100000001b3);
    }

    format!("{first:016x}{second:016x}")
}

fn unsupported(message: impl Into<String>) -> PyErr {
    RumbaUnsupportedError::new_err(message.into())
}

fn compilation(message: impl Into<String>) -> PyErr {
    RumbaCompilationError::new_err(message.into())
}
