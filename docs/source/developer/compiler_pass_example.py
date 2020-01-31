def ex_compiler_pass():

    # magictoken.ex_compiler_pass.begin
    from numba import njit
    from numba.core import ir
    from numba.compiler import CompilerBase, DefaultPassBuilder
    from numba.core.compiler_machinery import FunctionPass, register_pass
    from numba.core.untyped_passes import IRProcessing
    from numbers import Number

    # Register this pass with the compiler framework, declare that it will not
    # mutate the control flow graph and that it is not an analysis_only pass (it
    # potentially mutates the IR).
    @register_pass(mutates_CFG=False, analysis_only=False)
    class ConstsAddOne(FunctionPass):
        _name = "consts_add_one" # the common name for the pass

        def __init__(self):
            FunctionPass.__init__(self)

        # implement method to do the work, "state" is the internal compiler
        # state from the CompilerBase instance.
        def run_pass(self, state):
            func_ir = state.func_ir # get the FunctionIR object
            mutated = False # used to record whether this pass mutates the IR
            # walk the blocks
            for blk in func_ir.blocks.values():
                # find the assignment nodes in the block and walk them
                for assgn in blk.find_insts(ir.Assign):
                    # if an assignment value is a ir.Consts
                    if isinstance(assgn.value, ir.Const):
                        const_val = assgn.value
                        # if the value of the ir.Const is a Number
                        if isinstance(const_val.value, Number):
                            # then add one!
                            const_val.value += 1
                            mutated |= True
            return mutated # return True if the IR was mutated, False if not.
    # magictoken.ex_compiler_pass.end

    # magictoken.ex_compiler_defn.begin
    class MyCompiler(CompilerBase): # custom compiler extends from CompilerBase

        def define_pipelines(self):
            # define a new set of pipelines (just one in this case) and for ease
            # base it on an existing pipeline from the DefaultPassBuilder,
            # namely the "nopython" pipeline
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            # Add the new pass to run after IRProcessing
            pm.add_pass_after(ConstsAddOne, IRProcessing)
            # finalize
            pm.finalize()
            # return as an iterable, any number of pipelines may be defined!
            return [pm]
    # magictoken.ex_compiler_defn.end

    # magictoken.ex_compiler_call.begin
    @njit(pipeline_class=MyCompiler) # JIT compile using the custom compiler
    def foo(x):
        a = 10
        b = 20.2
        c = x + a + b
        return c

    print(foo(100)) # 100 + 10 + 20.2 (+ 1 + 1), extra + 1 + 1 from the rewrite!
    # magictoken.ex_compiler_call.end

    # magictoken.ex_compiler_timings.begin
    compile_result = foo.overloads[foo.signatures[0]]
    nopython_times = compile_result.metadata['pipeline_times']['nopython']
    for k in nopython_times.keys():
        if ConstsAddOne._name in k:
            print(nopython_times[k])
    # magictoken.ex_compiler_timings.end

    assert foo(100) == 132.2

ex_compiler_pass()
