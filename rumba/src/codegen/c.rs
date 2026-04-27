use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use crate::errors::unsupported;
use crate::ir::{ConstantValue, ExprNode, ParsedFunction, StmtNode, UnaryOp};
use crate::types::{promote_numeric, ScalarType};

struct CExpr {
    code: String,
    typ: ScalarType,
}

pub(crate) struct Emitter {
    function_ir: ParsedFunction,
    signature: Vec<ScalarType>,
    env: HashMap<String, ScalarType>,
    declared: HashSet<String>,
    lines: Vec<String>,
    return_type: Option<ScalarType>,
}

impl Emitter {
    pub(crate) fn new(function_ir: ParsedFunction, signature: Vec<ScalarType>) -> PyResult<Self> {
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

    pub(crate) fn emit(&mut self) -> PyResult<(String, ScalarType)> {
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

fn indent(level: usize) -> String {
    "    ".repeat(level)
}
