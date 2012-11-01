from llvm.core import InlineAsm, Type

_exp_f32_functype = Type.function(Type.float(), [Type.float()])
_exp_f32_ptx = '''{
.reg .pred 	%numbapro_p<3>;
.reg .f32 	%numbapro_f<16>;
.reg .s32 	%numbapro_r<2>;

mov.f32         %numbapro_f1, $1;
mul.f32         %numbapro_f2, %numbapro_f1, 0f3FB8AA3B;
cvt.rzi.f32.f32 %numbapro_f3, %numbapro_f2;
mov.f32         %numbapro_f4, 0fBF317200;
fma.rn.f32      %numbapro_f5, %numbapro_f3, %numbapro_f4, %numbapro_f1;
mov.f32         %numbapro_f6, 0fB5BFBE8E;
fma.rn.f32      %numbapro_f7, %numbapro_f3, %numbapro_f6, %numbapro_f5;
mul.f32         %numbapro_f8, %numbapro_f7, 0f3FB8AA3B;
ex2.approx.f32 	%numbapro_f9, %numbapro_f8;
add.f32         %numbapro_f10, %numbapro_f3, 0f00000000;
ex2.approx.f32 	%numbapro_f11, %numbapro_f10;
mul.f32         %numbapro_f12, %numbapro_f9, %numbapro_f11;
setp.lt.f32 	%numbapro_p1, %numbapro_f1, 0fC2D20000;
selp.f32        %numbapro_f13, 0f00000000, %numbapro_f12, %numbapro_p1;
mov.u32         %numbapro_r1, 2139095040;
mov.b32         %numbapro_f14, %numbapro_r1;
setp.gt.f32 	%numbapro_p2, %numbapro_f1, 0f42D20000;
selp.f32        %numbapro_f15, %numbapro_f14, %numbapro_f13, %numbapro_p2;
mov.f32         $0, %numbapro_f15;
}
'''
_exp_f32_constrains = '=f,f'

exp_f32 = InlineAsm.get(_exp_f32_functype, _exp_f32_ptx, _exp_f32_constrains)