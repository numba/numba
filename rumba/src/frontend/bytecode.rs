use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::errors::unsupported;
use crate::ir::{BinOp, CmpOp, ConstantValue, ExprNode, ParsedFunction, StmtNode, UnaryOp};

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct DecodedCodeObject {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) locals: Vec<String>,
    pub(crate) consts: Vec<Constant>,
    pub(crate) names: Vec<String>,
    pub(crate) bytecode: Vec<u8>,
    pub(crate) instructions: Vec<BytecodeInstruction>,
}

#[derive(Clone)]
pub(crate) enum Constant {
    None,
    Scalar(ConstantValue),
    Unsupported(String),
}

#[derive(Clone)]
pub(crate) struct BytecodeInstruction {
    pub(crate) offset: usize,
    pub(crate) opcode: Opcode,
    pub(crate) arg: Option<u32>,
    pub(crate) operand: Option<String>,
    pub(crate) starts_line: Option<usize>,
    size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opcode {
    LoadConst,
    LoadFast,
    StoreFast,
    LoadGlobal,
    BinaryOp,
    UnaryPositive,
    UnaryNegative,
    UnaryNot,
    CompareOp,
    ReturnValue,
    PopJumpIfFalse,
    PopJumpIfTrue,
    JumpForward,
    JumpBackward,
    Call,
    GetIter,
    ForIter,
    EndFor,
    PopTop,
    Unsupported(u16),
}

#[allow(dead_code)]
pub(crate) struct BasicBlock {
    pub(crate) id: usize,
    pub(crate) start_offset: usize,
    pub(crate) end_offset: usize,
    pub(crate) instructions: Vec<BytecodeInstruction>,
    pub(crate) terminator: Option<Opcode>,
}

#[derive(Clone)]
enum StackValue {
    Expr(ExprNode),
    Name(String),
    CallRange(Vec<ExprNode>),
    InplaceBinOp {
        left: ExprNode,
        op: BinOp,
        right: ExprNode,
    },
}

pub(crate) fn build_rumba_ast(
    _py: Python<'_>,
    func: &Bound<'_, PyAny>,
) -> PyResult<ParsedFunction> {
    parse_function(func, Vec::new())
}

pub(crate) fn inspect_bytecode(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let decoded = decode_function(func)?;
    let out = PyList::empty_bound(py);
    for inst in &decoded.instructions {
        let item = PyDict::new_bound(py);
        item.set_item("offset", inst.offset)?;
        item.set_item("opname", inst.opcode.name())?;
        item.set_item("arg", inst.arg)?;
        item.set_item("argrepr", inst.operand.as_deref().unwrap_or(""))?;
        item.set_item("starts_line", inst.starts_line)?;
        out.append(item)?;
    }
    Ok(out.into())
}

pub(crate) fn decode_function(func: &Bound<'_, PyAny>) -> PyResult<DecodedCodeObject> {
    reject_unsupported_function_shape(func)?;

    let code = func.getattr("__code__")?;
    let name = code.getattr("co_name")?.extract()?;
    let argcount: usize = code.getattr("co_argcount")?.extract()?;
    let varnames = extract_string_tuple(&code.getattr("co_varnames")?.downcast_into::<PyTuple>()?)?;
    let names = extract_string_tuple(&code.getattr("co_names")?.downcast_into::<PyTuple>()?)?;
    let consts = extract_consts(&code.getattr("co_consts")?.downcast_into::<PyTuple>()?)?;
    let bytecode = code
        .getattr("co_code")?
        .downcast_into::<PyBytes>()?
        .as_bytes()
        .to_vec();
    let line_starts = line_starts(&code)?;
    let instructions = decode_instructions(&bytecode, &consts, &names, &varnames, &line_starts)?;

    Ok(DecodedCodeObject {
        name,
        args: varnames.iter().take(argcount).cloned().collect(),
        locals: varnames,
        consts,
        names,
        bytecode,
        instructions,
    })
}

fn parse_function(
    func: &Bound<'_, PyAny>,
    mut call_stack: Vec<String>,
) -> PyResult<ParsedFunction> {
    let decoded = decode_function(func)?;
    if call_stack.iter().any(|name| name == &decoded.name) {
        return Err(unsupported(format!(
            "recursive function calls are not supported: {}",
            decoded.name
        )));
    }
    call_stack.push(decoded.name.clone());
    let globals = func.getattr("__globals__")?.downcast_into::<PyDict>()?;
    BytecodeParser::new(&decoded, &globals, call_stack).parse_function()
}

#[allow(dead_code)]
pub(crate) fn basic_blocks(code: &DecodedCodeObject) -> Vec<BasicBlock> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();
    let mut start = code
        .instructions
        .first()
        .map(|inst| inst.offset)
        .unwrap_or(0);
    for inst in &code.instructions {
        current.push(inst.clone());
        if inst.opcode.is_terminator() {
            blocks.push(BasicBlock {
                id: blocks.len(),
                start_offset: start,
                end_offset: inst.offset + inst.size,
                instructions: std::mem::take(&mut current),
                terminator: Some(inst.opcode),
            });
            start = inst.offset + inst.size;
        }
    }
    if !current.is_empty() {
        let end = current
            .last()
            .map(|inst| inst.offset + inst.size)
            .unwrap_or(start);
        blocks.push(BasicBlock {
            id: blocks.len(),
            start_offset: start,
            end_offset: end,
            instructions: current,
            terminator: None,
        });
    }
    blocks
}

struct BytecodeParser<'a> {
    code: &'a DecodedCodeObject,
    globals: &'a Bound<'a, PyDict>,
    call_stack: Vec<String>,
}

impl<'a> BytecodeParser<'a> {
    fn new(
        code: &'a DecodedCodeObject,
        globals: &'a Bound<'a, PyDict>,
        call_stack: Vec<String>,
    ) -> Self {
        Self {
            code,
            globals,
            call_stack,
        }
    }

    fn parse_function(&self) -> PyResult<ParsedFunction> {
        Ok(ParsedFunction {
            name: self.code.name.clone(),
            args: self.code.args.clone(),
            body: self.parse_block(0, self.code.instructions.len())?.0,
        })
    }

    fn parse_block(&self, start: usize, end: usize) -> PyResult<(Vec<StmtNode>, usize)> {
        let mut statements = Vec::new();
        let mut stack = Vec::new();
        let mut index = start;
        while index < end {
            let inst = &self.code.instructions[index];
            match inst.opcode {
                Opcode::LoadConst => stack.push(StackValue::Expr(self.const_expr(inst)?)),
                Opcode::LoadFast => {
                    stack.push(StackValue::Expr(ExprNode::Name(self.local_name(inst)?)))
                }
                Opcode::LoadGlobal => stack.push(StackValue::Name(self.name_operand(inst)?)),
                Opcode::StoreFast => {
                    let name = self.local_name(inst)?;
                    let value = pop_stack(&mut stack)?;
                    statements.push(store_stmt(name, value)?);
                }
                Opcode::BinaryOp => {
                    let right = pop_expr(&mut stack)?;
                    let left = pop_expr(&mut stack)?;
                    let (op, inplace) = binop(inst)?;
                    if inplace {
                        stack.push(StackValue::InplaceBinOp { left, op, right });
                    } else {
                        stack.push(StackValue::Expr(ExprNode::BinOp {
                            left: Box::new(left),
                            op,
                            right: Box::new(right),
                        }));
                    }
                }
                Opcode::UnaryPositive => {
                    let value = pop_expr(&mut stack)?;
                    stack.push(StackValue::Expr(ExprNode::UnaryOp {
                        op: UnaryOp::UAdd,
                        value: Box::new(value),
                    }));
                }
                Opcode::UnaryNegative => {
                    let value = pop_expr(&mut stack)?;
                    stack.push(StackValue::Expr(ExprNode::UnaryOp {
                        op: UnaryOp::USub,
                        value: Box::new(value),
                    }));
                }
                Opcode::UnaryNot => {
                    let value = pop_expr(&mut stack)?;
                    stack.push(StackValue::Expr(ExprNode::UnaryOp {
                        op: UnaryOp::Not,
                        value: Box::new(value),
                    }));
                }
                Opcode::CompareOp => {
                    let right = pop_expr(&mut stack)?;
                    let left = pop_expr(&mut stack)?;
                    stack.push(StackValue::Expr(ExprNode::Compare {
                        left: Box::new(left),
                        op: cmpop(inst)?,
                        right: Box::new(right),
                    }));
                }
                Opcode::Call => {
                    let argc = inst.arg.unwrap_or(0) as usize;
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc {
                        args.push(pop_expr(&mut stack)?);
                    }
                    args.reverse();
                    match pop_stack(&mut stack)? {
                        StackValue::Name(name) if name == "range" => {
                            stack.push(StackValue::CallRange(args));
                        }
                        StackValue::Name(name) => {
                            let function = self.resolve_global_function(&name)?;
                            stack.push(StackValue::Expr(ExprNode::Call {
                                function: Box::new(function),
                                args,
                            }));
                        }
                        _ => return Err(unsupported("unsupported function call")),
                    }
                }
                Opcode::GetIter => {}
                Opcode::ForIter => {
                    let range_args = match pop_stack(&mut stack)? {
                        StackValue::CallRange(args) => args,
                        _ => return Err(unsupported("only range(...) loops are supported")),
                    };
                    let target = self.jump_target(inst, true)?;
                    let store_index = index + 1;
                    if store_index >= end
                        || self.code.instructions[store_index].opcode != Opcode::StoreFast
                    {
                        return Err(unsupported(
                            "only simple range loop variables are supported",
                        ));
                    }
                    let loop_var = self.local_name(&self.code.instructions[store_index])?;
                    let jump_index = self.find_loop_jump(store_index + 1, target)?;
                    let (body, consumed) = self.parse_block(store_index + 1, jump_index)?;
                    if consumed != jump_index {
                        return Err(unsupported("unsupported control flow in range loop"));
                    }
                    let (start_expr, stop_expr, step_expr) = range_bounds(range_args)?;
                    statements.push(StmtNode::ForRange {
                        target: loop_var,
                        start: start_expr,
                        stop: stop_expr,
                        step: step_expr,
                        body,
                    });
                    index = self.index_at_or_after(target)?;
                    if index < end && self.code.instructions[index].opcode == Opcode::EndFor {
                        index += 1;
                    }
                    continue;
                }
                Opcode::PopJumpIfFalse | Opcode::PopJumpIfTrue => {
                    let test = pop_expr(&mut stack)?;
                    let target = self.jump_target(inst, true)?;
                    let target_index = self.index_at(target)?;
                    let body_start = index + 1;
                    if target_index > start {
                        if let Some(jump_index) =
                            self.trailing_forward_jump(body_start, target_index)?
                        {
                            let after =
                                self.jump_target(&self.code.instructions[jump_index], true)?;
                            let after_index = self.index_at(after)?;
                            let (body, body_end) = self.parse_block(body_start, jump_index)?;
                            let (orelse, else_end) = self.parse_block(target_index, after_index)?;
                            if body_end != jump_index || else_end != after_index {
                                return Err(unsupported("unsupported if/else control flow"));
                            }
                            statements.push(StmtNode::If {
                                test: maybe_invert_test(test, inst.opcode),
                                body,
                                orelse,
                            });
                            index = after_index;
                            continue;
                        }
                    }
                    let (body, body_end) = self.parse_block(body_start, target_index)?;
                    if body_end != target_index {
                        return Err(unsupported("unsupported if control flow"));
                    }
                    if body
                        .last()
                        .is_some_and(|stmt| matches!(stmt, StmtNode::Return(_)))
                        && target_index < end
                    {
                        let (orelse, else_end) = self.parse_block(target_index, end)?;
                        statements.push(StmtNode::If {
                            test: maybe_invert_test(test, inst.opcode),
                            body,
                            orelse,
                        });
                        index = else_end;
                        continue;
                    }
                    statements.push(StmtNode::If {
                        test: maybe_invert_test(test, inst.opcode),
                        body,
                        orelse: Vec::new(),
                    });
                    index = target_index;
                    continue;
                }
                Opcode::ReturnValue => {
                    statements.push(StmtNode::Return(pop_expr(&mut stack)?));
                    return Ok((statements, index + 1));
                }
                Opcode::JumpForward | Opcode::JumpBackward => return Ok((statements, index)),
                Opcode::EndFor | Opcode::PopTop => {}
                Opcode::Unsupported(opcode) => {
                    return Err(unsupported(format!("unsupported bytecode opcode {opcode}")));
                }
            }
            index += 1;
        }
        Ok((statements, index))
    }

    fn const_expr(&self, inst: &BytecodeInstruction) -> PyResult<ExprNode> {
        let index = inst.arg.unwrap_or(0) as usize;
        match self.code.consts.get(index) {
            Some(Constant::Scalar(value)) => Ok(ExprNode::Constant(value.clone())),
            Some(Constant::None) => Err(unsupported("None constants are not supported")),
            Some(Constant::Unsupported(name)) => {
                Err(unsupported(format!("unsupported constant {name}")))
            }
            None => Err(unsupported("constant index out of range")),
        }
    }

    fn local_name(&self, inst: &BytecodeInstruction) -> PyResult<String> {
        self.code
            .locals
            .get(inst.arg.unwrap_or(0) as usize)
            .cloned()
            .ok_or_else(|| unsupported("local variable index out of range"))
    }

    fn name_operand(&self, inst: &BytecodeInstruction) -> PyResult<String> {
        global_name_index(inst.arg.unwrap_or(0))
            .and_then(|index| self.code.names.get(index).cloned())
            .ok_or_else(|| unsupported("name index out of range"))
    }

    fn jump_target(&self, inst: &BytecodeInstruction, forward: bool) -> PyResult<usize> {
        let arg = inst
            .arg
            .ok_or_else(|| unsupported("jump instruction missing target"))?
            as usize;
        if forward {
            Ok(inst.offset + inst.size + arg * 2)
        } else {
            Ok(inst.offset + inst.size - arg * 2)
        }
    }

    fn index_at(&self, offset: usize) -> PyResult<usize> {
        self.code
            .instructions
            .iter()
            .position(|inst| inst.offset == offset)
            .ok_or_else(|| unsupported("jump target does not align with decoded instruction"))
    }

    fn index_at_or_after(&self, offset: usize) -> PyResult<usize> {
        self.code
            .instructions
            .iter()
            .position(|inst| inst.offset >= offset)
            .ok_or_else(|| unsupported("jump target is outside decoded bytecode"))
    }

    fn find_loop_jump(&self, start: usize, target: usize) -> PyResult<usize> {
        for index in start..self.code.instructions.len() {
            let inst = &self.code.instructions[index];
            if inst.offset >= target {
                break;
            }
            if inst.opcode == Opcode::JumpBackward
                && self.jump_target(inst, false)? <= self.code.instructions[start - 1].offset
            {
                return Ok(index);
            }
        }
        Err(unsupported("unsupported range loop control flow"))
    }

    fn trailing_forward_jump(&self, start: usize, end: usize) -> PyResult<Option<usize>> {
        if start >= end {
            return Ok(None);
        }
        let jump_index = end - 1;
        if self.code.instructions[jump_index].opcode == Opcode::JumpForward {
            Ok(Some(jump_index))
        } else {
            Ok(None)
        }
    }

    fn resolve_global_function(&self, name: &str) -> PyResult<ParsedFunction> {
        let Some(value) = self.globals.get_item(name)? else {
            return Err(unsupported(format!("unsupported call to {name}")));
        };
        if !value.hasattr("__code__")? {
            return Err(unsupported(format!("unsupported call to {name}")));
        }
        parse_function(&value, self.call_stack.clone())
    }
}

fn reject_unsupported_function_shape(func: &Bound<'_, PyAny>) -> PyResult<()> {
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
    if !func.getattr("__defaults__")?.is_none() || !func.getattr("__kwdefaults__")?.is_none() {
        return Err(unsupported("default arguments are not supported"));
    }
    let flags: u32 = code.getattr("co_flags")?.extract()?;
    if flags & 0x0c != 0 {
        return Err(unsupported("varargs and kwargs are not supported"));
    }
    if flags & 0x20 != 0 {
        return Err(unsupported("generators are not supported"));
    }
    let exception_table = code
        .getattr("co_exceptiontable")?
        .downcast_into::<PyBytes>()?;
    if !exception_table.as_bytes().is_empty() {
        return Err(unsupported("exception handling is not supported"));
    }
    Ok(())
}

fn extract_string_tuple(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<String>> {
    tuple.iter().map(|item| item.extract()).collect()
}

fn extract_consts(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<Constant>> {
    tuple
        .iter()
        .map(|item| {
            if item.is_none() {
                Ok(Constant::None)
            } else if let Ok(value) = item.extract::<bool>() {
                Ok(Constant::Scalar(ConstantValue::Bool(value)))
            } else if let Ok(value) = item.extract::<i64>() {
                Ok(Constant::Scalar(ConstantValue::Int(value)))
            } else if let Ok(value) = item.extract::<f64>() {
                Ok(Constant::Scalar(ConstantValue::Float(value)))
            } else {
                Ok(Constant::Unsupported(item.get_type().name()?.to_string()))
            }
        })
        .collect()
}

fn line_starts(code: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    if let Ok(lines) = code.call_method0("co_lines") {
        for line in lines.iter()? {
            let line = line?.downcast_into::<PyTuple>()?;
            if line.len() == 3 {
                if let Some(lineno) = line.get_item(2)?.extract::<Option<usize>>()? {
                    out.push((line.get_item(0)?.extract()?, lineno));
                }
            }
        }
    }
    Ok(out)
}

fn decode_instructions(
    bytecode: &[u8],
    consts: &[Constant],
    names: &[String],
    locals: &[String],
    line_starts: &[(usize, usize)],
) -> PyResult<Vec<BytecodeInstruction>> {
    let mut out = Vec::new();
    let mut offset = 0;
    let mut extended_arg = 0_u32;
    while offset + 1 < bytecode.len() {
        let raw_opcode = bytecode[offset] as u16;
        let raw_arg = bytecode[offset + 1] as u32;
        let opcode = opcode_from_raw(raw_opcode);
        let arg = if opcode.has_arg(raw_opcode) {
            Some((extended_arg << 8) | raw_arg)
        } else {
            None
        };
        if raw_opcode == 144 {
            extended_arg = arg.unwrap_or(0);
            offset += 2;
            continue;
        }
        extended_arg = 0;
        let size = 2 + inline_cache_entries(opcode) * 2;
        if !opcode.is_bookkeeping() {
            out.push(BytecodeInstruction {
                offset,
                opcode,
                arg,
                operand: operand_repr(opcode, arg, consts, names, locals),
                starts_line: line_starts
                    .iter()
                    .find_map(|(line_offset, line)| (*line_offset == offset).then_some(*line)),
                size,
            });
        }
        offset += size;
    }
    Ok(out)
}

fn opcode_from_raw(opcode: u16) -> Opcode {
    match opcode {
        1 => Opcode::PopTop,
        4 => Opcode::EndFor,
        10 => Opcode::UnaryPositive,
        11 => Opcode::UnaryNegative,
        12 => Opcode::UnaryNot,
        68 => Opcode::GetIter,
        83 => Opcode::ReturnValue,
        93 => Opcode::ForIter,
        100 => Opcode::LoadConst,
        107 => Opcode::CompareOp,
        110 => Opcode::JumpForward,
        114 => Opcode::PopJumpIfFalse,
        115 => Opcode::PopJumpIfTrue,
        116 => Opcode::LoadGlobal,
        122 => Opcode::BinaryOp,
        124 => Opcode::LoadFast,
        125 => Opcode::StoreFast,
        134 | 140 => Opcode::JumpBackward,
        171 => Opcode::Call,
        other => Opcode::Unsupported(other),
    }
}

fn inline_cache_entries(opcode: Opcode) -> usize {
    match opcode {
        Opcode::BinaryOp | Opcode::CompareOp | Opcode::ForIter => 1,
        Opcode::LoadGlobal => 4,
        Opcode::Call => 3,
        _ => 0,
    }
}

fn operand_repr(
    opcode: Opcode,
    arg: Option<u32>,
    consts: &[Constant],
    names: &[String],
    locals: &[String],
) -> Option<String> {
    let arg = arg? as usize;
    match opcode {
        Opcode::LoadConst => consts.get(arg).map(Constant::repr),
        Opcode::LoadFast | Opcode::StoreFast => locals.get(arg).cloned(),
        Opcode::LoadGlobal => {
            global_name_index(arg as u32).and_then(|index| names.get(index).cloned())
        }
        Opcode::BinaryOp => Some(binary_op_name(arg as u32).to_string()),
        Opcode::CompareOp => Some(compare_op_name(arg as u32).to_string()),
        _ => Some(arg.to_string()),
    }
}

fn global_name_index(arg: u32) -> Option<usize> {
    Some((arg >> 1) as usize)
}

fn binop(inst: &BytecodeInstruction) -> PyResult<(BinOp, bool)> {
    match inst.arg.unwrap_or(0) {
        0 => Ok((BinOp::Add, false)),
        2 => Ok((BinOp::FloorDiv, false)),
        5 => Ok((BinOp::Mult, false)),
        6 => Ok((BinOp::Mod, false)),
        10 => Ok((BinOp::Sub, false)),
        11 => Ok((BinOp::Div, false)),
        13 => Ok((BinOp::Add, true)),
        15 => Ok((BinOp::FloorDiv, true)),
        18 => Ok((BinOp::Mult, true)),
        19 => Ok((BinOp::Mod, true)),
        23 => Ok((BinOp::Sub, true)),
        24 => Ok((BinOp::Div, true)),
        _ => Err(unsupported("unsupported binary operator")),
    }
}

fn cmpop(inst: &BytecodeInstruction) -> PyResult<CmpOp> {
    match inst.arg.unwrap_or(0) & 0x0f {
        0 => Ok(CmpOp::Lt),
        1 => Ok(CmpOp::LtE),
        2 => Ok(CmpOp::Eq),
        3 => Ok(CmpOp::NotEq),
        4 => Ok(CmpOp::Gt),
        5 => Ok(CmpOp::GtE),
        _ => Err(unsupported("unsupported comparison operator")),
    }
}

fn binary_op_name(arg: u32) -> &'static str {
    match arg {
        0 => "+",
        2 => "//",
        5 => "*",
        6 => "%",
        10 => "-",
        11 => "/",
        13 => "+=",
        15 => "//=",
        18 => "*=",
        19 => "%=",
        23 => "-=",
        24 => "/=",
        _ => "<unsupported>",
    }
}

fn compare_op_name(arg: u32) -> &'static str {
    match arg & 0x0f {
        0 => "<",
        1 => "<=",
        2 => "==",
        3 => "!=",
        4 => ">",
        5 => ">=",
        _ => "<unsupported>",
    }
}

fn pop_stack(stack: &mut Vec<StackValue>) -> PyResult<StackValue> {
    stack
        .pop()
        .ok_or_else(|| unsupported("unsupported stack effect in bytecode"))
}

fn pop_expr(stack: &mut Vec<StackValue>) -> PyResult<ExprNode> {
    match pop_stack(stack)? {
        StackValue::Expr(expr) => Ok(expr),
        StackValue::InplaceBinOp { left, op, right } => Ok(ExprNode::BinOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }),
        _ => Err(unsupported("expected scalar expression on bytecode stack")),
    }
}

fn store_stmt(name: String, value: StackValue) -> PyResult<StmtNode> {
    match value {
        StackValue::InplaceBinOp { left, op, right } => {
            if let ExprNode::Name(left_name) = &left {
                if left_name == &name {
                    return Ok(StmtNode::AugAssign {
                        name,
                        op,
                        value: right,
                    });
                }
            }
            Ok(StmtNode::Assign {
                name,
                value: ExprNode::BinOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
            })
        }
        StackValue::Expr(value) => Ok(StmtNode::Assign { name, value }),
        _ => Err(unsupported("unsupported store target bytecode")),
    }
}

fn range_bounds(args: Vec<ExprNode>) -> PyResult<(ExprNode, ExprNode, ExprNode)> {
    match args.len() {
        1 => Ok((
            ExprNode::Constant(ConstantValue::Int(0)),
            args[0].clone(),
            ExprNode::Constant(ConstantValue::Int(1)),
        )),
        2 => Ok((
            args[0].clone(),
            args[1].clone(),
            ExprNode::Constant(ConstantValue::Int(1)),
        )),
        3 => Ok((args[0].clone(), args[1].clone(), args[2].clone())),
        _ => Err(unsupported("range accepts one to three arguments")),
    }
}

fn maybe_invert_test(test: ExprNode, opcode: Opcode) -> ExprNode {
    if opcode == Opcode::PopJumpIfTrue {
        ExprNode::UnaryOp {
            op: UnaryOp::Not,
            value: Box::new(test),
        }
    } else {
        test
    }
}

impl Constant {
    fn repr(&self) -> String {
        match self {
            Constant::None => "None".to_string(),
            Constant::Scalar(ConstantValue::Bool(value)) => value.to_string(),
            Constant::Scalar(ConstantValue::Int(value)) => value.to_string(),
            Constant::Scalar(ConstantValue::Float(value)) => value.to_string(),
            Constant::Unsupported(name) => format!("<{name}>"),
        }
    }
}

impl Opcode {
    fn name(self) -> &'static str {
        match self {
            Opcode::LoadConst => "LOAD_CONST",
            Opcode::LoadFast => "LOAD_FAST",
            Opcode::StoreFast => "STORE_FAST",
            Opcode::LoadGlobal => "LOAD_GLOBAL",
            Opcode::BinaryOp => "BINARY_OP",
            Opcode::UnaryPositive => "UNARY_POSITIVE",
            Opcode::UnaryNegative => "UNARY_NEGATIVE",
            Opcode::UnaryNot => "UNARY_NOT",
            Opcode::CompareOp => "COMPARE_OP",
            Opcode::ReturnValue => "RETURN_VALUE",
            Opcode::PopJumpIfFalse => "POP_JUMP_IF_FALSE",
            Opcode::PopJumpIfTrue => "POP_JUMP_IF_TRUE",
            Opcode::JumpForward => "JUMP_FORWARD",
            Opcode::JumpBackward => "JUMP_BACKWARD",
            Opcode::Call => "CALL",
            Opcode::GetIter => "GET_ITER",
            Opcode::ForIter => "FOR_ITER",
            Opcode::EndFor => "END_FOR",
            Opcode::PopTop => "POP_TOP",
            Opcode::Unsupported(_) => "UNSUPPORTED",
        }
    }

    fn has_arg(self, raw: u16) -> bool {
        raw >= 90
    }

    fn is_bookkeeping(self) -> bool {
        matches!(self, Opcode::Unsupported(0 | 9 | 151))
    }

    fn is_terminator(self) -> bool {
        matches!(
            self,
            Opcode::ReturnValue
                | Opcode::PopJumpIfFalse
                | Opcode::PopJumpIfTrue
                | Opcode::JumpForward
                | Opcode::JumpBackward
                | Opcode::ForIter
        )
    }
}
