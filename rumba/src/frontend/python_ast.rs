use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::errors::unsupported;
use crate::frontend::{class_name, getattr};
use crate::ir::{BinOp, CmpOp, ConstantValue, ExprNode, ParsedFunction, StmtNode, UnaryOp};

pub(crate) fn build_rumba_ast(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<ParsedFunction> {
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
