def extract_shape_arg(shape_argref):
    if not isinstance(shape_argref, tuple):
        assert shape_argref.value.kind == 'Const'
        if isinstance(shape_argref.value.args.value, tuple):
            shape = shape_argref.value.args.value
            assert all(isinstance(x, int) for x in shape)
            return shape

        shape_argref = (shape_argref,)


    shape_args = tuple(x.value for x in shape_argref)
    for sarg in shape_args:
        if sarg.kind != 'Const':
            msg = "shape must be a constant"
            raise CudaPyInferError(sarg, msg)
    shape = tuple(x.args.value for x in shape_args)
    return shape
