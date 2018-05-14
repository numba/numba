from llvmlite import ir
from llvmlite.ir import PointerType as ptr, LiteralStructType as struct

i8, i16, i32, i64 = map(ir.IntType, [8, 16, 32, 64])
index = lambda i: ir.Constant(i32, i)

context = ir.global_context

ndt_slice_t = context.get_identified_type("ndt_slice_t")
ndt_slice_t.set_body(i64, i64, i64)

ndt_t = context.get_identified_type("_ndt")
ndt_t.set_body(
    i32, i32, i32, i32, i64, i16,
    struct([struct([i64, i64, i64, ptr(ptr(ndt_t))])]),
    struct([struct([struct([i32, i64, i32, ptr(i32), i32, ptr(ndt_slice_t)])])]),
    ir.ArrayType(i8, 16)
)

xnd_bitmap_t = context.get_identified_type("xnd_bitmap")
xnd_bitmap_t.set_body(
    ptr(i8),
    i64,
    ptr(xnd_bitmap_t)
)

xnd_t = context.get_identified_type("xnd")
xnd_t.set_body(
    xnd_bitmap_t,
    i64,
    ptr(ndt_t),
    ptr(i8)
)

ndt_context_t = context.get_identified_type("_ndt_context_t")
ndt_context_t.set_body(
    i32, i32, i32,
    struct([ptr(i8)])
)
