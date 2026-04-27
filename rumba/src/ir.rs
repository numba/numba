pub(crate) struct ParsedFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) body: Vec<StmtNode>,
}

pub(crate) enum StmtNode {
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
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Return(_) => "Return",
            Self::Assign { .. } => "Assign",
            Self::AugAssign { .. } => "AugAssign",
            Self::If { .. } => "If",
            Self::ForRange { .. } => "For",
        }
    }
}

pub(crate) enum ExprNode {
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

pub(crate) enum ConstantValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Copy)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
}

impl BinOp {
    pub(crate) fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mult => "*",
            Self::Div => "/",
            Self::FloorDiv => "/",
            Self::Mod => "%",
        }
    }

    pub(crate) fn type_name(self) -> &'static str {
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

pub(crate) enum UnaryOp {
    Not,
    USub,
    UAdd,
}

pub(crate) enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
}

impl CmpOp {
    pub(crate) fn symbol(&self) -> &'static str {
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
