'''
    
Constrains:
    "h" = .u16 reg
    "r" = .u32 reg
    "l" = .u64 reg
    "f" = .f32 reg
    "d" = .f64 reg
'''
from llvm.core import InlineAsm, Type

# exp()

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

_exp_f64_functype = Type.function(Type.double(), [Type.double()])
_exp_f64_ptx = '''{
.reg .pred 	%numbapro_p<8>;
.reg .s32 	%numbapro_r<14>;
.reg .s64 	%numbapro_rd<2>;
.reg .f64 	%numbapro_fd<43>;


mov.f64 	%numbapro_fd6, $1;
{
.reg .b32 %temp;
mov.b64 	{%temp, %numbapro_r1}, %numbapro_fd6;
}
setp.lt.u32 	%numbapro_p1, %numbapro_r1, 1082535491;
setp.lt.s32 	%numbapro_p2, %numbapro_r1, -1064875759;
or.pred  	%numbapro_p3, %numbapro_p1, %numbapro_p2;
@%numbapro_p3 bra 	BB1_2;

setp.lt.s32 	%numbapro_p4, %numbapro_r1, 0;
mov.u64 	%numbapro_rd1, 9218868437227405312;
mov.b64 	%numbapro_fd7, %numbapro_rd1;
selp.f64 	%numbapro_fd8, 0d0000000000000000, %numbapro_fd7, %numbapro_p4;
abs.f64 	%numbapro_fd9, %numbapro_fd6;
setp.gtu.f64 	%numbapro_p5, %numbapro_fd9, %numbapro_fd7;
add.f64 	%numbapro_fd10, %numbapro_fd6, %numbapro_fd6;
selp.f64 	%numbapro_fd42, %numbapro_fd10, %numbapro_fd8, %numbapro_p5;
bra.uni 	BB1_5;

BB1_2:
mul.f64 	%numbapro_fd11, %numbapro_fd6, 0d3FF71547652B82FE;
cvt.rni.f64.f64 	%numbapro_fd12, %numbapro_fd11;
cvt.rzi.s32.f64 	%numbapro_r2, %numbapro_fd12;
mov.f64 	%numbapro_fd13, 0dBFE62E42FEFA39EF;
fma.rn.f64 	%numbapro_fd14, %numbapro_fd12, %numbapro_fd13, %numbapro_fd6;
mov.f64 	%numbapro_fd15, 0dBC7ABC9E3B39803F;
fma.rn.f64 	%numbapro_fd16, %numbapro_fd12, %numbapro_fd15, %numbapro_fd14;
mov.f64 	%numbapro_fd17, 0d3E928A27E30F5561;
mov.f64 	%numbapro_fd18, 0d3E5AE6449C0686C0;
fma.rn.f64 	%numbapro_fd19, %numbapro_fd18, %numbapro_fd16, %numbapro_fd17;
mov.f64 	%numbapro_fd20, 0d3EC71DE8E6486D6B;
fma.rn.f64 	%numbapro_fd21, %numbapro_fd19, %numbapro_fd16, %numbapro_fd20;
mov.f64 	%numbapro_fd22, 0d3EFA019A6B2464C5;
fma.rn.f64 	%numbapro_fd23, %numbapro_fd21, %numbapro_fd16, %numbapro_fd22;
mov.f64 	%numbapro_fd24, 0d3F2A01A0171064A5;
fma.rn.f64 	%numbapro_fd25, %numbapro_fd23, %numbapro_fd16, %numbapro_fd24;
mov.f64 	%numbapro_fd26, 0d3F56C16C17F29C8D;
fma.rn.f64 	%numbapro_fd27, %numbapro_fd25, %numbapro_fd16, %numbapro_fd26;
mov.f64 	%numbapro_fd28, 0d3F8111111111A24E;
fma.rn.f64 	%numbapro_fd29, %numbapro_fd27, %numbapro_fd16, %numbapro_fd28;
mov.f64 	%numbapro_fd30, 0d3FA555555555211D;
fma.rn.f64 	%numbapro_fd31, %numbapro_fd29, %numbapro_fd16, %numbapro_fd30;
mov.f64 	%numbapro_fd32, 0d3FC5555555555530;
fma.rn.f64 	%numbapro_fd33, %numbapro_fd31, %numbapro_fd16, %numbapro_fd32;
mov.f64 	%numbapro_fd34, 0d3FE0000000000005;
fma.rn.f64 	%numbapro_fd35, %numbapro_fd33, %numbapro_fd16, %numbapro_fd34;
mov.f64 	%numbapro_fd36, 0d3FF0000000000000;
fma.rn.f64 	%numbapro_fd37, %numbapro_fd35, %numbapro_fd16, %numbapro_fd36;
fma.rn.f64 	%numbapro_fd2, %numbapro_fd37, %numbapro_fd16, %numbapro_fd36;
shl.b32 	%numbapro_r3, %numbapro_r2, 20;
add.s32 	%numbapro_r4, %numbapro_r3, 1072693248;
abs.s32 	%numbapro_r5, %numbapro_r2;
setp.lt.s32 	%numbapro_p6, %numbapro_r5, 1021;
@%numbapro_p6 bra 	BB1_4;

add.s32 	%numbapro_r6, %numbapro_r3, 1130364928;
setp.lt.s32 	%numbapro_p7, %numbapro_r2, 0;
mov.u32 	%numbapro_r7, 0;
selp.b32 	%numbapro_r8, %numbapro_r6, %numbapro_r4, %numbapro_p7;
shr.s32 	%numbapro_r9, %numbapro_r2, 31;
add.s32 	%numbapro_r10, %numbapro_r9, 1073741824;
and.b32  	%numbapro_r11, %numbapro_r10, -57671680;
add.s32 	%numbapro_r12, %numbapro_r8, -1048576;
mov.b64 	%numbapro_fd38, {%numbapro_r7, %numbapro_r11};
mul.f64 	%numbapro_fd39, %numbapro_fd2, %numbapro_fd38;
mov.b64 	%numbapro_fd40, {%numbapro_r7, %numbapro_r12};
mul.f64 	%numbapro_fd42, %numbapro_fd39, %numbapro_fd40;
bra.uni 	BB1_5;

BB1_4:
mov.u32 	%numbapro_r13, 0;
mov.b64 	%numbapro_fd41, {%numbapro_r13, %numbapro_r4};
mul.f64 	%numbapro_fd42, %numbapro_fd41, %numbapro_fd2;

BB1_5:
mov.f64     $0, %numbapro_fd42;
}
'''
_exp_f64_constrains = '=d,d'
exp_f64 = InlineAsm.get(_exp_f64_functype, _exp_f64_ptx, _exp_f64_constrains)

# fabs()

_fabs_f32_functype = Type.function(Type.float(), [Type.float()])
_fabs_f32_ptx = 'abs.f32 $0, $1;'
_fabs_f32_contrains = '=f,f'
fabs_f32 = InlineAsm.get(_fabs_f32_functype, _fabs_f32_ptx, _fabs_f32_contrains)

_fabs_f64_functype = Type.function(Type.double(), [Type.double()])
_fabs_f64_ptx = 'abs.f64 $0, $1;'
_fabs_f64_contrains = '=d,d'
fabs_f64 = InlineAsm.get(_fabs_f64_functype, _fabs_f64_ptx, _fabs_f64_contrains)

# log()

_log_f32_functype = Type.function(Type.float(), [Type.float()])
_log_f32_ptx = '''{
.reg .pred 	%numbapro_p<6>;
.reg .f32 	%numbapro_f<29>;
.reg .s32 	%numbapro_r<12>;


mov.f32 	%numbapro_f4, $1;
mov.b32 	%numbapro_r1, %numbapro_f4;
mov.u32 	%numbapro_r2, 2139095040;
mov.b32 	%numbapro_f5, %numbapro_r2;
setp.gt.f32 	%numbapro_p1, %numbapro_f5, %numbapro_f4;
setp.gt.f32 	%numbapro_p2, %numbapro_f4, 0f00000000;
and.pred  	%numbapro_p3, %numbapro_p2, %numbapro_p1;
@%numbapro_p3 bra 	BB0_2;

lg2.approx.f32 	%numbapro_f6, %numbapro_f4;
mul.f32 	%numbapro_f28, %numbapro_f6, 0f3F317218;
bra.uni 	BB0_3;

BB0_2:
setp.lt.u32 	%numbapro_p4, %numbapro_r1, 8388608;
mul.f32 	%numbapro_f7, %numbapro_f4, 0f4B800000;
mov.b32 	%numbapro_r3, %numbapro_f7;
selp.b32 	%numbapro_r4, %numbapro_r3, %numbapro_r1, %numbapro_p4;
selp.b32 	%numbapro_r5, -151, -127, %numbapro_p4;
and.b32  	%numbapro_r6, %numbapro_r4, -2139095041;
or.b32  	%numbapro_r7, %numbapro_r6, 1065353216;
mov.b32 	%numbapro_f8, %numbapro_r7;
shr.u32 	%numbapro_r8, %numbapro_r4, 23;
setp.gt.f32 	%numbapro_p5, %numbapro_f8, 0f3FB504F3;
mul.f32 	%numbapro_f9, %numbapro_f8, 0f3F000000;
selp.f32 	%numbapro_f10, %numbapro_f9, %numbapro_f8, %numbapro_p5;
selp.u32 	%numbapro_r9, 1, 0, %numbapro_p5;
add.s32 	%numbapro_r10, %numbapro_r8, %numbapro_r5;
add.s32 	%numbapro_r11, %numbapro_r10, %numbapro_r9;
add.f32 	%numbapro_f11, %numbapro_f10, 0fBF800000;
add.f32 	%numbapro_f12, %numbapro_f10, 0f3F800000;
div.approx.f32 	%numbapro_f13, %numbapro_f11, %numbapro_f12;
neg.f32 	%numbapro_f14, %numbapro_f11;
mul.rn.f32 	%numbapro_f15, %numbapro_f14, %numbapro_f13;
add.rn.f32 	%numbapro_f16, %numbapro_f11, %numbapro_f15;
mul.f32 	%numbapro_f17, %numbapro_f16, %numbapro_f16;
mov.f32 	%numbapro_f18, 0f3C4C4BE0;
mov.f32 	%numbapro_f19, 0f3B2063C3;
fma.rn.f32 	%numbapro_f20, %numbapro_f19, %numbapro_f17, %numbapro_f18;
mov.f32 	%numbapro_f21, 0f3DAAAB50;
fma.rn.f32 	%numbapro_f22, %numbapro_f20, %numbapro_f17, %numbapro_f21;
mul.f32 	%numbapro_f23, %numbapro_f22, %numbapro_f17;
fma.rn.f32 	%numbapro_f24, %numbapro_f23, %numbapro_f16, %numbapro_f15;
add.f32 	%numbapro_f25, %numbapro_f24, %numbapro_f11;
cvt.rn.f32.s32 	%numbapro_f26, %numbapro_r11;
mov.f32 	%numbapro_f27, 0f3F317218;
fma.rn.f32 	%numbapro_f28, %numbapro_f26, %numbapro_f27, %numbapro_f25;

BB0_3:
mov.f32	$0, %numbapro_f28;
}
'''
_log_f32_constrains = '=f,f'
log_f32 = InlineAsm.get(_log_f32_functype, _log_f32_ptx, _log_f32_constrains)

_log_f64_functype = Type.function(Type.double(), [Type.double()])
_log_f64_ptx = '''{
.reg .pred 	%numbapro_p<9>;
.reg .f32 	%numbapro_f<5>;
.reg .s32 	%numbapro_r<24>;
.reg .s64 	%numbapro_rd<3>;
.reg .f64 	%numbapro_fd<58>;


mov.f64 	%numbapro_fd10, $1;
{
.reg .b32 %temp;
mov.b64 	{%temp, %numbapro_r20}, %numbapro_fd10;
}
{
.reg .b32 %temp;
mov.b64 	{%numbapro_r21, %temp}, %numbapro_fd10;
}
mov.u64 	%numbapro_rd1, 9218868437227405312;
mov.b64 	%numbapro_fd1, %numbapro_rd1;
setp.gt.f64 	%numbapro_p1, %numbapro_fd1, %numbapro_fd10;
setp.gt.f64 	%numbapro_p2, %numbapro_fd10, 0d0000000000000000;
and.pred  	%numbapro_p3, %numbapro_p2, %numbapro_p1;
@%numbapro_p3 bra 	BB1_6;

abs.f64 	%numbapro_fd11, %numbapro_fd10;
setp.gtu.f64 	%numbapro_p4, %numbapro_fd11, %numbapro_fd1;
@%numbapro_p4 bra 	BB1_5;

setp.eq.f64 	%numbapro_p5, %numbapro_fd10, 0d0000000000000000;
@%numbapro_p5 bra 	BB1_4;

setp.eq.f64 	%numbapro_p6, %numbapro_fd1, %numbapro_fd10;
mov.u64 	%numbapro_rd2, -2251799813685248;
mov.b64 	%numbapro_fd12, %numbapro_rd2;
selp.f64 	%numbapro_fd57, %numbapro_fd10, %numbapro_fd12, %numbapro_p6;
bra.uni 	BB1_11;

BB1_4:
neg.f64 	%numbapro_fd57, %numbapro_fd1;
bra.uni 	BB1_11;

BB1_5:
add.f64 	%numbapro_fd57, %numbapro_fd10, %numbapro_fd10;
bra.uni 	BB1_11;

BB1_6:
setp.gt.u32 	%numbapro_p7, %numbapro_r20, 1048575;
mov.u32 	%numbapro_r22, -1023;
@%numbapro_p7 bra 	BB1_8;

mul.f64 	%numbapro_fd13, %numbapro_fd10, 0d4350000000000000;
{
.reg .b32 %temp;
mov.b64 	{%temp, %numbapro_r20}, %numbapro_fd13;
}
{
.reg .b32 %temp;
mov.b64 	{%numbapro_r21, %temp}, %numbapro_fd13;
}
mov.u32 	%numbapro_r22, -1077;

BB1_8:
shr.s32 	%numbapro_r13, %numbapro_r20, 20;
add.s32 	%numbapro_r23, %numbapro_r22, %numbapro_r13;
and.b32  	%numbapro_r14, %numbapro_r20, -2146435073;
or.b32  	%numbapro_r15, %numbapro_r14, 1072693248;
mov.b64 	%numbapro_fd56, {%numbapro_r21, %numbapro_r15};
setp.lt.u32 	%numbapro_p8, %numbapro_r15, 1073127583;
@%numbapro_p8 bra 	BB1_10;

{
.reg .b32 %temp;
mov.b64 	{%numbapro_r16, %temp}, %numbapro_fd56;
}
{
.reg .b32 %temp;
mov.b64 	{%temp, %numbapro_r17}, %numbapro_fd56;
}
add.s32 	%numbapro_r18, %numbapro_r17, -1048576;
mov.b64 	%numbapro_fd56, {%numbapro_r16, %numbapro_r18};
add.s32 	%numbapro_r23, %numbapro_r23, 1;

BB1_10:
add.f64 	%numbapro_fd14, %numbapro_fd56, 0d3FF0000000000000;
mov.f64 	%numbapro_fd16, 0d3FF0000000000000;
// inline asm
cvt.rn.f32.f64     %numbapro_f1,%numbapro_fd14;
// inline asm
// inline asm
rcp.approx.f32.ftz %numbapro_f2,%numbapro_f1;
// inline asm
// inline asm
cvt.f64.f32        %numbapro_fd15,%numbapro_f2;
// inline asm
neg.f64 	%numbapro_fd17, %numbapro_fd14;
fma.rn.f64 	%numbapro_fd18, %numbapro_fd17, %numbapro_fd15, %numbapro_fd16;
fma.rn.f64 	%numbapro_fd19, %numbapro_fd18, %numbapro_fd18, %numbapro_fd18;
fma.rn.f64 	%numbapro_fd20, %numbapro_fd19, %numbapro_fd15, %numbapro_fd15;
add.f64 	%numbapro_fd21, %numbapro_fd56, 0dBFF0000000000000;
mul.f64 	%numbapro_fd22, %numbapro_fd21, %numbapro_fd20;
fma.rn.f64 	%numbapro_fd23, %numbapro_fd21, %numbapro_fd20, %numbapro_fd22;
mul.f64 	%numbapro_fd24, %numbapro_fd23, %numbapro_fd23;
mov.f64 	%numbapro_fd25, 0d3ED0EE258B7A8B04;
mov.f64 	%numbapro_fd26, 0d3EB1380B3AE80F1E;
fma.rn.f64 	%numbapro_fd27, %numbapro_fd26, %numbapro_fd24, %numbapro_fd25;
mov.f64 	%numbapro_fd28, 0d3EF3B2669F02676F;
fma.rn.f64 	%numbapro_fd29, %numbapro_fd27, %numbapro_fd24, %numbapro_fd28;
mov.f64 	%numbapro_fd30, 0d3F1745CBA9AB0956;
fma.rn.f64 	%numbapro_fd31, %numbapro_fd29, %numbapro_fd24, %numbapro_fd30;
mov.f64 	%numbapro_fd32, 0d3F3C71C72D1B5154;
fma.rn.f64 	%numbapro_fd33, %numbapro_fd31, %numbapro_fd24, %numbapro_fd32;
mov.f64 	%numbapro_fd34, 0d3F624924923BE72D;
fma.rn.f64 	%numbapro_fd35, %numbapro_fd33, %numbapro_fd24, %numbapro_fd34;
mov.f64 	%numbapro_fd36, 0d3F8999999999A3C4;
fma.rn.f64 	%numbapro_fd37, %numbapro_fd35, %numbapro_fd24, %numbapro_fd36;
mov.f64 	%numbapro_fd38, 0d3FB5555555555554;
fma.rn.f64 	%numbapro_fd39, %numbapro_fd37, %numbapro_fd24, %numbapro_fd38;
sub.f64 	%numbapro_fd40, %numbapro_fd21, %numbapro_fd23;
add.f64 	%numbapro_fd41, %numbapro_fd40, %numbapro_fd40;
neg.f64 	%numbapro_fd42, %numbapro_fd23;
fma.rn.f64 	%numbapro_fd43, %numbapro_fd42, %numbapro_fd21, %numbapro_fd41;
mul.f64 	%numbapro_fd44, %numbapro_fd20, %numbapro_fd43;
mul.f64 	%numbapro_fd45, %numbapro_fd39, %numbapro_fd24;
fma.rn.f64 	%numbapro_fd46, %numbapro_fd45, %numbapro_fd23, %numbapro_fd44;
cvt.rn.f64.s32 	%numbapro_fd47, %numbapro_r23;
mov.f64 	%numbapro_fd48, 0d3FE62E42FEFA39EF;
fma.rn.f64 	%numbapro_fd49, %numbapro_fd47, %numbapro_fd48, %numbapro_fd23;
neg.s32 	%numbapro_r19, %numbapro_r23;
cvt.rn.f64.s32 	%numbapro_fd50, %numbapro_r19;
fma.rn.f64 	%numbapro_fd51, %numbapro_fd50, %numbapro_fd48, %numbapro_fd49;
sub.f64 	%numbapro_fd52, %numbapro_fd51, %numbapro_fd23;
sub.f64 	%numbapro_fd53, %numbapro_fd46, %numbapro_fd52;
mov.f64 	%numbapro_fd54, 0d3C7ABC9E3B39803F;
fma.rn.f64 	%numbapro_fd55, %numbapro_fd47, %numbapro_fd54, %numbapro_fd53;
add.f64 	%numbapro_fd57, %numbapro_fd49, %numbapro_fd55;

BB1_11:
mov.f64	$0, %numbapro_fd57;
}
'''
_log_f64_constrains = '=d,d'
log_f64 = InlineAsm.get(_log_f64_functype, _log_f64_ptx, _log_f64_constrains)

# pow()

_pow_f32_functype = Type.function(Type.float(), [Type.float(), Type.float()])
_pow_f32_ptx = '''{
    .reg .pred 	%numbapro_p<30>;
	.reg .f32 	%numbapro_f<122>;
	.reg .s32 	%numbapro_r<28>;


	mov.f32 	%numbapro_f118, $1;
	mov.f32 	%numbapro_f26, $2;
	setp.eq.f32 	%numbapro_p1, %numbapro_f118, 0f3F800000;
	mov.f32 	%numbapro_f121, 0f3F800000;
	setp.eq.f32 	%numbapro_p2, %numbapro_f26, 0f00000000;
	or.pred  	%numbapro_p3, %numbapro_p1, %numbapro_p2;
	@%numbapro_p3 bra 	BB0_25;

	abs.f32 	%numbapro_f1, %numbapro_f118;
	mov.u32 	%numbapro_r1, 2139095040;
	mov.b32 	%numbapro_f2, %numbapro_r1;
	setp.gtu.f32 	%numbapro_p4, %numbapro_f1, %numbapro_f2;
	@%numbapro_p4 bra 	BB0_24;

	abs.f32 	%numbapro_f3, %numbapro_f26;
	setp.gtu.f32 	%numbapro_p5, %numbapro_f3, %numbapro_f2;
	@%numbapro_p5 bra 	BB0_24;

	setp.eq.f32 	%numbapro_p6, %numbapro_f2, %numbapro_f118;
	@%numbapro_p6 bra 	BB0_23;

	setp.eq.f32 	%numbapro_p7, %numbapro_f3, %numbapro_f2;
	@%numbapro_p7 bra 	BB0_19;

	mul.f32 	%numbapro_f28, %numbapro_f26, 0f3F000000;
	cvt.rzi.f32.f32 	%numbapro_f29, %numbapro_f28;
	fma.rn.f32 	%numbapro_f30, %numbapro_f29, 0fC0000000, %numbapro_f26;
	abs.f32 	%numbapro_f4, %numbapro_f30;
	setp.eq.f32 	%numbapro_p8, %numbapro_f118, 0f00000000;
	@%numbapro_p8 bra 	BB0_16;

	neg.f32 	%numbapro_f31, %numbapro_f2;
	setp.eq.f32 	%numbapro_p9, %numbapro_f118, %numbapro_f31;
	@%numbapro_p9 bra 	BB0_13;

	setp.geu.f32 	%numbapro_p10, %numbapro_f118, 0f00000000;
	@%numbapro_p10 bra 	BB0_9;

	cvt.rzi.f32.f32 	%numbapro_f32, %numbapro_f26;
	setp.neu.f32 	%numbapro_p11, %numbapro_f32, %numbapro_f26;
	@%numbapro_p11 bra 	BB0_12;

    BB0_9:
	mov.b32 	%numbapro_r2, %numbapro_f1;
	shr.u32 	%numbapro_r3, %numbapro_r2, 23;
	and.b32  	%numbapro_r4, %numbapro_r3, 255;
	setp.eq.s32 	%numbapro_p12, %numbapro_r4, 0;
	mul.f32 	%numbapro_f33, %numbapro_f1, 0f4B800000;
	mov.b32 	%numbapro_r5, %numbapro_f33;
	shr.u32 	%numbapro_r6, %numbapro_r5, 23;
	and.b32  	%numbapro_r7, %numbapro_r6, 255;
	add.s32 	%numbapro_r8, %numbapro_r7, -24;
	selp.f32 	%numbapro_f34, %numbapro_f33, %numbapro_f1, %numbapro_p12;
	selp.b32 	%numbapro_r9, %numbapro_r8, %numbapro_r4, %numbapro_p12;
	mov.b32 	%numbapro_r10, %numbapro_f34;
	and.b32  	%numbapro_r11, %numbapro_r10, -2139095041;
	or.b32  	%numbapro_r12, %numbapro_r11, 1065353216;
	mov.b32 	%numbapro_f35, %numbapro_r12;
	setp.gt.f32 	%numbapro_p13, %numbapro_f35, 0f3FB504F3;
	mul.f32 	%numbapro_f36, %numbapro_f35, 0f3F000000;
	selp.b32 	%numbapro_r13, -126, -127, %numbapro_p13;
	add.s32 	%numbapro_r14, %numbapro_r9, %numbapro_r13;
	selp.f32 	%numbapro_f37, %numbapro_f36, %numbapro_f35, %numbapro_p13;
	add.f32 	%numbapro_f38, %numbapro_f37, 0fBF800000;
	add.f32 	%numbapro_f39, %numbapro_f37, 0f3F800000;
	rcp.rn.f32 	%numbapro_f40, %numbapro_f39;
	add.f32 	%numbapro_f41, %numbapro_f38, %numbapro_f38;
	mul.f32 	%numbapro_f42, %numbapro_f41, %numbapro_f40;
	mul.f32 	%numbapro_f43, %numbapro_f42, %numbapro_f42;
	mov.f32 	%numbapro_f44, 0f3C4CAF63;
	mov.f32 	%numbapro_f45, 0f3B18F0FE;
	fma.rn.f32 	%numbapro_f46, %numbapro_f45, %numbapro_f43, %numbapro_f44;
	mov.f32 	%numbapro_f47, 0f3DAAAABD;
	fma.rn.f32 	%numbapro_f48, %numbapro_f46, %numbapro_f43, %numbapro_f47;
	mul.rn.f32 	%numbapro_f49, %numbapro_f48, %numbapro_f43;
	mul.rn.f32 	%numbapro_f50, %numbapro_f49, %numbapro_f42;
	mov.b32 	%numbapro_r15, %numbapro_f42;
	and.b32  	%numbapro_r16, %numbapro_r15, -4096;
	mov.b32 	%numbapro_f51, %numbapro_r16;
	mov.b32 	%numbapro_r17, %numbapro_f38;
	and.b32  	%numbapro_r18, %numbapro_r17, -4096;
	mov.b32 	%numbapro_f52, %numbapro_r18;
	sub.f32 	%numbapro_f53, %numbapro_f38, %numbapro_f51;
	add.f32 	%numbapro_f54, %numbapro_f53, %numbapro_f53;
	sub.f32 	%numbapro_f55, %numbapro_f38, %numbapro_f52;
	neg.f32 	%numbapro_f56, %numbapro_f51;
	fma.rn.f32 	%numbapro_f57, %numbapro_f56, %numbapro_f52, %numbapro_f54;
	fma.rn.f32 	%numbapro_f58, %numbapro_f56, %numbapro_f55, %numbapro_f57;
	mul.rn.f32 	%numbapro_f59, %numbapro_f40, %numbapro_f58;
	add.f32 	%numbapro_f60, %numbapro_f51, %numbapro_f59;
	sub.f32 	%numbapro_f61, %numbapro_f60, %numbapro_f51;
	sub.f32 	%numbapro_f62, %numbapro_f59, %numbapro_f61;
	add.f32 	%numbapro_f63, %numbapro_f60, %numbapro_f50;
	sub.f32 	%numbapro_f64, %numbapro_f60, %numbapro_f63;
	add.f32 	%numbapro_f65, %numbapro_f64, %numbapro_f50;
	add.f32 	%numbapro_f66, %numbapro_f65, %numbapro_f62;
	add.f32 	%numbapro_f67, %numbapro_f63, %numbapro_f66;
	sub.f32 	%numbapro_f68, %numbapro_f63, %numbapro_f67;
	add.f32 	%numbapro_f69, %numbapro_f68, %numbapro_f66;
	cvt.rn.f32.s32 	%numbapro_f70, %numbapro_r14;
	mov.f32 	%numbapro_f71, 0f3F317200;
	mul.rn.f32 	%numbapro_f72, %numbapro_f70, %numbapro_f71;
	mov.f32 	%numbapro_f73, 0f35BFBE8E;
	mul.rn.f32 	%numbapro_f74, %numbapro_f70, %numbapro_f73;
	add.f32 	%numbapro_f75, %numbapro_f72, %numbapro_f67;
	sub.f32 	%numbapro_f76, %numbapro_f72, %numbapro_f75;
	add.f32 	%numbapro_f77, %numbapro_f76, %numbapro_f67;
	add.f32 	%numbapro_f78, %numbapro_f77, %numbapro_f69;
	add.f32 	%numbapro_f79, %numbapro_f78, %numbapro_f74;
	add.f32 	%numbapro_f80, %numbapro_f75, %numbapro_f79;
	sub.f32 	%numbapro_f81, %numbapro_f75, %numbapro_f80;
	add.f32 	%numbapro_f82, %numbapro_f81, %numbapro_f79;
	mul.f32 	%numbapro_f83, %numbapro_f26, 0f39000000;
	setp.gt.f32 	%numbapro_p14, %numbapro_f3, 0f77F684DF;
	selp.f32 	%numbapro_f84, %numbapro_f83, %numbapro_f26, %numbapro_p14;
	mul.rn.f32 	%numbapro_f85, %numbapro_f84, %numbapro_f80;
	neg.f32 	%numbapro_f86, %numbapro_f85;
	fma.rn.f32 	%numbapro_f87, %numbapro_f84, %numbapro_f80, %numbapro_f86;
	fma.rn.f32 	%numbapro_f88, %numbapro_f84, %numbapro_f82, %numbapro_f87;
	mov.f32 	%numbapro_f89, 0f00000000;
	fma.rn.f32 	%numbapro_f90, %numbapro_f89, %numbapro_f80, %numbapro_f88;
	add.rn.f32 	%numbapro_f91, %numbapro_f85, %numbapro_f90;
	neg.f32 	%numbapro_f92, %numbapro_f91;
	add.rn.f32 	%numbapro_f93, %numbapro_f85, %numbapro_f92;
	add.rn.f32 	%numbapro_f94, %numbapro_f93, %numbapro_f90;
	mov.b32 	%numbapro_r19, %numbapro_f91;
	setp.eq.s32 	%numbapro_p15, %numbapro_r19, 1118925336;
	add.s32 	%numbapro_r20, %numbapro_r19, -1;
	mov.b32 	%numbapro_f95, %numbapro_r20;
	mov.u32 	%numbapro_r21, 922746880;
	mov.b32 	%numbapro_f96, %numbapro_r21;
	add.f32 	%numbapro_f97, %numbapro_f94, %numbapro_f96;
	selp.f32 	%numbapro_f5, %numbapro_f97, %numbapro_f94, %numbapro_p15;
	selp.f32 	%numbapro_f98, %numbapro_f95, %numbapro_f91, %numbapro_p15;
	mul.f32 	%numbapro_f99, %numbapro_f98, 0f3FB8AA3B;
	cvt.rzi.f32.f32 	%numbapro_f100, %numbapro_f99;
	mov.f32 	%numbapro_f101, 0fBF317200;
	fma.rn.f32 	%numbapro_f102, %numbapro_f100, %numbapro_f101, %numbapro_f98;
	mov.f32 	%numbapro_f103, 0fB5BFBE8E;
	fma.rn.f32 	%numbapro_f104, %numbapro_f100, %numbapro_f103, %numbapro_f102;
	mul.f32 	%numbapro_f105, %numbapro_f104, 0f3FB8AA3B;
	ex2.approx.f32 	%numbapro_f106, %numbapro_f105;
	add.f32 	%numbapro_f107, %numbapro_f100, 0f00000000;
	ex2.approx.f32 	%numbapro_f108, %numbapro_f107;
	mul.f32 	%numbapro_f109, %numbapro_f106, %numbapro_f108;
	setp.lt.f32 	%numbapro_p16, %numbapro_f98, 0fC2D20000;
	selp.f32 	%numbapro_f110, 0f00000000, %numbapro_f109, %numbapro_p16;
	setp.gt.f32 	%numbapro_p17, %numbapro_f98, 0f42D20000;
	selp.f32 	%numbapro_f117, %numbapro_f2, %numbapro_f110, %numbapro_p17;
	setp.eq.f32 	%numbapro_p18, %numbapro_f117, %numbapro_f2;
	@%numbapro_p18 bra 	BB0_11;

	fma.rn.f32 	%numbapro_f117, %numbapro_f117, %numbapro_f5, %numbapro_f117;

    BB0_11:
	mov.f32 	%numbapro_f116, $0;
	setp.eq.f32 	%numbapro_p19, %numbapro_f4, 0f3F800000;
	setp.lt.f32 	%numbapro_p20, %numbapro_f116, 0f00000000;
	and.pred  	%numbapro_p21, %numbapro_p20, %numbapro_p19;
	mov.b32 	%numbapro_r22, %numbapro_f117;
	xor.b32  	%numbapro_r23, %numbapro_r22, -2147483648;
	mov.b32 	%numbapro_f111, %numbapro_r23;
	selp.f32 	%numbapro_f121, %numbapro_f111, %numbapro_f117, %numbapro_p21;
	bra.uni 	BB0_25;

    BB0_12:
	mov.u32 	%numbapro_r24, -4194304;
	mov.b32 	%numbapro_f112, %numbapro_r24;
	rsqrt.approx.f32 	%numbapro_f121, %numbapro_f112;
	bra.uni 	BB0_25;

    BB0_13:
	setp.geu.f32 	%numbapro_p22, %numbapro_f26, 0f00000000;
	@%numbapro_p22 bra 	BB0_15;

	rcp.rn.f32 	%numbapro_f118, %numbapro_f118;

    BB0_15:
	neg.f32 	%numbapro_f113, %numbapro_f118;
	mov.b32 	%numbapro_r25, %numbapro_f113;
	xor.b32  	%numbapro_r26, %numbapro_r25, -2147483648;
	mov.b32 	%numbapro_f114, %numbapro_r26;
	setp.eq.f32 	%numbapro_p23, %numbapro_f4, 0f3F800000;
	selp.f32 	%numbapro_f121, %numbapro_f114, %numbapro_f113, %numbapro_p23;
	bra.uni 	BB0_25;

    BB0_16:
	setp.eq.f32 	%numbapro_p24, %numbapro_f4, 0f3F800000;
	selp.f32 	%numbapro_f119, %numbapro_f118, 0f00000000, %numbapro_p24;
	setp.geu.f32 	%numbapro_p25, %numbapro_f26, 0f00000000;
	@%numbapro_p25 bra 	BB0_18;

	rcp.rn.f32 	%numbapro_f119, %numbapro_f119;

    BB0_18:
	add.f32 	%numbapro_f121, %numbapro_f119, %numbapro_f119;
	bra.uni 	BB0_25;

    BB0_19:
	setp.eq.f32 	%numbapro_p26, %numbapro_f118, 0fBF800000;
	mov.f32 	%numbapro_f121, 0f3F800000;
	@%numbapro_p26 bra 	BB0_25;

	setp.gt.f32 	%numbapro_p27, %numbapro_f1, 0f3F800000;
	selp.f32 	%numbapro_f120, %numbapro_f2, 0f00000000, %numbapro_p27;
	setp.geu.f32 	%numbapro_p28, %numbapro_f26, 0f00000000;
	@%numbapro_p28 bra 	BB0_22;

	rcp.rn.f32 	%numbapro_f120, %numbapro_f120;

    BB0_22:
	add.f32 	%numbapro_f121, %numbapro_f120, %numbapro_f120;
	bra.uni 	BB0_25;

    BB0_23:
	mov.b32 	%numbapro_r27, %numbapro_f26;
	setp.lt.s32 	%numbapro_p29, %numbapro_r27, 0;
	selp.f32 	%numbapro_f121, 0f00000000, %numbapro_f2, %numbapro_p29;
	bra.uni 	BB0_25;
    
    BB0_24:
	add.f32 	%numbapro_f121, %numbapro_f118, %numbapro_f26;
    
    BB0_25:
	mov.f32	$0, %numbapro_f121;
}
'''
_pow_f32_constrains = '=f,f,f'
pow_f32 = InlineAsm.get(_pow_f32_functype, _pow_f32_ptx, _pow_f32_constrains)


_pow_f64_functype = Type.function(Type.double(), [Type.double(), Type.double()])
_pow_f64_ptx = '''{
	.reg .pred 	%numbapro_p<34>;
	.reg .f32 	%numbapro_f<5>;
	.reg .s32 	%numbapro_r<39>;
	.reg .s64 	%numbapro_rd<7>;
	.reg .f64 	%numbapro_fd<174>;


	mov.f64 	%numbapro_fd30, $1;
	mov.f64 	%numbapro_fd31, $2;
	setp.eq.f64 	%numbapro_p1, %numbapro_fd30, 0d3FF0000000000000;
	mov.f64 	%numbapro_fd173, 0d3FF0000000000000;
	setp.eq.f64 	%numbapro_p2, %numbapro_fd31, 0d0000000000000000;
	or.pred  	%numbapro_p3, %numbapro_p1, %numbapro_p2;
	@%numbapro_p3 bra 	BB1_33;

	abs.f64 	%numbapro_fd1, %numbapro_fd30;
	mov.u64 	%numbapro_rd1, 9218868437227405312;
	mov.b64 	%numbapro_fd2, %numbapro_rd1;
	setp.gtu.f64 	%numbapro_p4, %numbapro_fd1, %numbapro_fd2;
	@%numbapro_p4 bra 	BB1_32;

	abs.f64 	%numbapro_fd3, %numbapro_fd31;
	setp.gtu.f64 	%numbapro_p5, %numbapro_fd3, %numbapro_fd2;
	@%numbapro_p5 bra 	BB1_32;

	setp.eq.f64 	%numbapro_p6, %numbapro_fd2, %numbapro_fd30;
	@%numbapro_p6 bra 	BB1_31;

	setp.eq.f64 	%numbapro_p7, %numbapro_fd3, %numbapro_fd2;
	@%numbapro_p7 bra 	BB1_28;

	mul.f64 	%numbapro_fd33, %numbapro_fd31, 0d3FE0000000000000;
	cvt.rzi.f64.f64 	%numbapro_fd34, %numbapro_fd33;
	fma.rn.f64 	%numbapro_fd35, %numbapro_fd34, 0dC000000000000000, %numbapro_fd31;
	abs.f64 	%numbapro_fd4, %numbapro_fd35;
	setp.eq.f64 	%numbapro_p8, %numbapro_fd30, 0d0000000000000000;
	@%numbapro_p8 bra 	BB1_26;

	neg.f64 	%numbapro_fd36, %numbapro_fd2;
	setp.eq.f64 	%numbapro_p9, %numbapro_fd30, %numbapro_fd36;
	@%numbapro_p9 bra 	BB1_22;

	setp.geu.f64 	%numbapro_p10, %numbapro_fd30, 0d0000000000000000;
	@%numbapro_p10 bra 	BB1_9;

	cvt.rzi.f64.f64 	%numbapro_fd37, %numbapro_fd31;
	setp.neu.f64 	%numbapro_p11, %numbapro_fd37, %numbapro_fd31;
	@%numbapro_p11 bra 	BB1_21;

    BB1_9:
	{
	.reg .b32 %temp;
	mov.b64 	{%temp, %numbapro_r36}, %numbapro_fd1;
	}
	{
	.reg .b32 %temp;
	mov.b64 	{%numbapro_r35, %temp}, %numbapro_fd1;
	}
	shr.u32 	%numbapro_r17, %numbapro_r36, 20;
	and.b32  	%numbapro_r37, %numbapro_r17, 2047;
	setp.ne.s32 	%numbapro_p12, %numbapro_r37, 0;
	@%numbapro_p12 bra 	BB1_11;

	mul.f64 	%numbapro_fd38, %numbapro_fd1, 0d4350000000000000;
	{
	.reg .b32 %temp;
	mov.b64 	{%temp, %numbapro_r36}, %numbapro_fd38;
	}
	{
	.reg .b32 %temp;
	mov.b64 	{%numbapro_r35, %temp}, %numbapro_fd38;
	}
	shr.u32 	%numbapro_r18, %numbapro_r36, 20;
	and.b32  	%numbapro_r19, %numbapro_r18, 2047;
	add.s32 	%numbapro_r37, %numbapro_r19, -54;

    BB1_11:
	add.s32 	%numbapro_r38, %numbapro_r37, -1023;
	and.b32  	%numbapro_r20, %numbapro_r36, -2146435073;
	or.b32  	%numbapro_r21, %numbapro_r20, 1072693248;
	mov.b64 	%numbapro_fd170, {%numbapro_r35, %numbapro_r21};
	setp.lt.u32 	%numbapro_p13, %numbapro_r21, 1073127583;
	@%numbapro_p13 bra 	BB1_13;

	{
	.reg .b32 %temp;
	mov.b64 	{%numbapro_r22, %temp}, %numbapro_fd170;
	}
	{
	.reg .b32 %temp;
	mov.b64 	{%temp, %numbapro_r23}, %numbapro_fd170;
	}
	add.s32 	%numbapro_r24, %numbapro_r23, -1048576;
	mov.b64 	%numbapro_fd170, {%numbapro_r22, %numbapro_r24};
	add.s32 	%numbapro_r38, %numbapro_r37, -1022;

    BB1_13:
	add.f64 	%numbapro_fd39, %numbapro_fd170, 0d3FF0000000000000;
	mov.f64 	%numbapro_fd41, 0d3FF0000000000000;
	// inline asm
	cvt.rn.f32.f64     %numbapro_f1,%numbapro_fd39;
	// inline asm
	// inline asm
	rcp.approx.f32.ftz %numbapro_f2,%numbapro_f1;
	// inline asm
	// inline asm
	cvt.f64.f32        %numbapro_fd40,%numbapro_f2;
	// inline asm
	neg.f64 	%numbapro_fd42, %numbapro_fd39;
	fma.rn.f64 	%numbapro_fd43, %numbapro_fd42, %numbapro_fd40, %numbapro_fd41;
	fma.rn.f64 	%numbapro_fd44, %numbapro_fd43, %numbapro_fd43, %numbapro_fd43;
	fma.rn.f64 	%numbapro_fd45, %numbapro_fd44, %numbapro_fd40, %numbapro_fd40;
	add.f64 	%numbapro_fd46, %numbapro_fd170, 0dBFF0000000000000;
	mul.f64 	%numbapro_fd47, %numbapro_fd46, %numbapro_fd45;
	fma.rn.f64 	%numbapro_fd48, %numbapro_fd46, %numbapro_fd45, %numbapro_fd47;
	mul.f64 	%numbapro_fd49, %numbapro_fd48, %numbapro_fd48;
	mov.f64 	%numbapro_fd50, 0d3ED0F5D241AD3B5A;
	mov.f64 	%numbapro_fd51, 0d3EB0F5FF7D2CAFE2;
	fma.rn.f64 	%numbapro_fd52, %numbapro_fd51, %numbapro_fd49, %numbapro_fd50;
	mov.f64 	%numbapro_fd53, 0d3EF3B20A75488A3F;
	fma.rn.f64 	%numbapro_fd54, %numbapro_fd52, %numbapro_fd49, %numbapro_fd53;
	mov.f64 	%numbapro_fd55, 0d3F1745CDE4FAECD5;
	fma.rn.f64 	%numbapro_fd56, %numbapro_fd54, %numbapro_fd49, %numbapro_fd55;
	mov.f64 	%numbapro_fd57, 0d3F3C71C7258A578B;
	fma.rn.f64 	%numbapro_fd58, %numbapro_fd56, %numbapro_fd49, %numbapro_fd57;
	mov.f64 	%numbapro_fd59, 0d3F6249249242B910;
	fma.rn.f64 	%numbapro_fd60, %numbapro_fd58, %numbapro_fd49, %numbapro_fd59;
	mov.f64 	%numbapro_fd61, 0d3F89999999999DFB;
	fma.rn.f64 	%numbapro_fd62, %numbapro_fd60, %numbapro_fd49, %numbapro_fd61;
	sub.f64 	%numbapro_fd63, %numbapro_fd46, %numbapro_fd48;
	add.f64 	%numbapro_fd64, %numbapro_fd63, %numbapro_fd63;
	neg.f64 	%numbapro_fd65, %numbapro_fd48;
	fma.rn.f64 	%numbapro_fd66, %numbapro_fd65, %numbapro_fd46, %numbapro_fd64;
	mul.f64 	%numbapro_fd67, %numbapro_fd45, %numbapro_fd66;
	mov.f64 	%numbapro_fd68, 0d3FB5555555555555;
	fma.rn.f64 	%numbapro_fd69, %numbapro_fd62, %numbapro_fd49, 0d3FB5555555555555;
	sub.f64 	%numbapro_fd70, %numbapro_fd68, %numbapro_fd69;
	fma.rn.f64 	%numbapro_fd71, %numbapro_fd62, %numbapro_fd49, %numbapro_fd70;
	add.f64 	%numbapro_fd72, %numbapro_fd71, 0d0000000000000000;
	add.f64 	%numbapro_fd73, %numbapro_fd72, 0dBC46A4CB00B9E7B0;
	add.f64 	%numbapro_fd74, %numbapro_fd69, %numbapro_fd73;
	sub.f64 	%numbapro_fd75, %numbapro_fd69, %numbapro_fd74;
	add.f64 	%numbapro_fd76, %numbapro_fd75, %numbapro_fd73;
	mul.rn.f64 	%numbapro_fd77, %numbapro_fd74, %numbapro_fd48;
	neg.f64 	%numbapro_fd78, %numbapro_fd77;
	fma.rn.f64 	%numbapro_fd79, %numbapro_fd74, %numbapro_fd48, %numbapro_fd78;
	fma.rn.f64 	%numbapro_fd80, %numbapro_fd74, %numbapro_fd67, %numbapro_fd79;
	fma.rn.f64 	%numbapro_fd81, %numbapro_fd76, %numbapro_fd48, %numbapro_fd80;
	add.f64 	%numbapro_fd82, %numbapro_fd77, %numbapro_fd81;
	sub.f64 	%numbapro_fd83, %numbapro_fd77, %numbapro_fd82;
	add.f64 	%numbapro_fd84, %numbapro_fd83, %numbapro_fd81;
	mul.rn.f64 	%numbapro_fd85, %numbapro_fd82, %numbapro_fd48;
	neg.f64 	%numbapro_fd86, %numbapro_fd85;
	fma.rn.f64 	%numbapro_fd87, %numbapro_fd82, %numbapro_fd48, %numbapro_fd86;
	fma.rn.f64 	%numbapro_fd88, %numbapro_fd82, %numbapro_fd67, %numbapro_fd87;
	fma.rn.f64 	%numbapro_fd89, %numbapro_fd84, %numbapro_fd48, %numbapro_fd88;
	add.f64 	%numbapro_fd90, %numbapro_fd85, %numbapro_fd89;
	sub.f64 	%numbapro_fd91, %numbapro_fd85, %numbapro_fd90;
	add.f64 	%numbapro_fd92, %numbapro_fd91, %numbapro_fd89;
	mul.rn.f64 	%numbapro_fd93, %numbapro_fd90, %numbapro_fd48;
	neg.f64 	%numbapro_fd94, %numbapro_fd93;
	fma.rn.f64 	%numbapro_fd95, %numbapro_fd90, %numbapro_fd48, %numbapro_fd94;
	fma.rn.f64 	%numbapro_fd96, %numbapro_fd90, %numbapro_fd67, %numbapro_fd95;
	fma.rn.f64 	%numbapro_fd97, %numbapro_fd92, %numbapro_fd48, %numbapro_fd96;
	add.f64 	%numbapro_fd98, %numbapro_fd93, %numbapro_fd97;
	sub.f64 	%numbapro_fd99, %numbapro_fd93, %numbapro_fd98;
	add.f64 	%numbapro_fd100, %numbapro_fd99, %numbapro_fd97;
	add.f64 	%numbapro_fd101, %numbapro_fd48, %numbapro_fd98;
	sub.f64 	%numbapro_fd102, %numbapro_fd48, %numbapro_fd101;
	add.f64 	%numbapro_fd103, %numbapro_fd102, %numbapro_fd98;
	add.f64 	%numbapro_fd104, %numbapro_fd103, %numbapro_fd100;
	fma.rn.f64 	%numbapro_fd105, %numbapro_fd45, %numbapro_fd66, %numbapro_fd104;
	add.f64 	%numbapro_fd106, %numbapro_fd101, %numbapro_fd105;
	sub.f64 	%numbapro_fd107, %numbapro_fd101, %numbapro_fd106;
	add.f64 	%numbapro_fd108, %numbapro_fd107, %numbapro_fd105;
	cvt.rn.f64.s32 	%numbapro_fd109, %numbapro_r38;
	mov.f64 	%numbapro_fd110, 0d3FE62E42FEFA3000;
	mul.rn.f64 	%numbapro_fd111, %numbapro_fd109, %numbapro_fd110;
	mov.f64 	%numbapro_fd112, 0d3D53DE6AF278ECE6;
	mul.rn.f64 	%numbapro_fd113, %numbapro_fd109, %numbapro_fd112;
	add.f64 	%numbapro_fd114, %numbapro_fd111, %numbapro_fd106;
	sub.f64 	%numbapro_fd115, %numbapro_fd111, %numbapro_fd114;
	add.f64 	%numbapro_fd116, %numbapro_fd115, %numbapro_fd106;
	add.f64 	%numbapro_fd117, %numbapro_fd116, %numbapro_fd108;
	add.f64 	%numbapro_fd118, %numbapro_fd117, %numbapro_fd113;
	add.f64 	%numbapro_fd119, %numbapro_fd114, %numbapro_fd118;
	sub.f64 	%numbapro_fd120, %numbapro_fd114, %numbapro_fd119;
	add.f64 	%numbapro_fd121, %numbapro_fd120, %numbapro_fd118;
	mul.f64 	%numbapro_fd122, %numbapro_fd31, 0d3F20000000000000;
	setp.gt.f64 	%numbapro_p14, %numbapro_fd3, 0d7F0D2A1BE4048F90;
	selp.f64 	%numbapro_fd123, %numbapro_fd122, %numbapro_fd31, %numbapro_p14;
	mul.rn.f64 	%numbapro_fd124, %numbapro_fd119, %numbapro_fd123;
	neg.f64 	%numbapro_fd125, %numbapro_fd124;
	fma.rn.f64 	%numbapro_fd126, %numbapro_fd119, %numbapro_fd123, %numbapro_fd125;
	fma.rn.f64 	%numbapro_fd127, %numbapro_fd121, %numbapro_fd123, %numbapro_fd126;
	add.f64 	%numbapro_fd8, %numbapro_fd124, %numbapro_fd127;
	sub.f64 	%numbapro_fd128, %numbapro_fd124, %numbapro_fd8;
	add.f64 	%numbapro_fd9, %numbapro_fd128, %numbapro_fd127;
	{
	.reg .b32 %temp;
	mov.b64 	{%temp, %numbapro_r13}, %numbapro_fd8;
	}
	setp.lt.u32 	%numbapro_p15, %numbapro_r13, 1082535491;
	setp.lt.s32 	%numbapro_p16, %numbapro_r13, -1064875759;
	or.pred  	%numbapro_p17, %numbapro_p15, %numbapro_p16;
	@%numbapro_p17 bra 	BB1_15;

	setp.lt.s32 	%numbapro_p18, %numbapro_r13, 0;
	selp.f64 	%numbapro_fd129, 0d0000000000000000, %numbapro_fd2, %numbapro_p18;
	abs.f64 	%numbapro_fd130, %numbapro_fd8;
	setp.gtu.f64 	%numbapro_p19, %numbapro_fd130, %numbapro_fd2;
	add.f64 	%numbapro_fd131, %numbapro_fd8, %numbapro_fd8;
	selp.f64 	%numbapro_fd171, %numbapro_fd131, %numbapro_fd129, %numbapro_p19;
	bra.uni 	BB1_18;

    BB1_15:
	mov.f64 	%numbapro_fd169, 0d3FF0000000000000;
	mul.f64 	%numbapro_fd132, %numbapro_fd8, 0d3FF71547652B82FE;
	cvt.rni.f64.f64 	%numbapro_fd133, %numbapro_fd132;
	cvt.rzi.s32.f64 	%numbapro_r14, %numbapro_fd133;
	mov.f64 	%numbapro_fd134, 0dBFE62E42FEFA39EF;
	fma.rn.f64 	%numbapro_fd135, %numbapro_fd133, %numbapro_fd134, %numbapro_fd8;
	mov.f64 	%numbapro_fd136, 0dBC7ABC9E3B39803F;
	fma.rn.f64 	%numbapro_fd137, %numbapro_fd133, %numbapro_fd136, %numbapro_fd135;
	mov.f64 	%numbapro_fd138, 0d3E928A27E30F5561;
	mov.f64 	%numbapro_fd139, 0d3E5AE6449C0686C0;
	fma.rn.f64 	%numbapro_fd140, %numbapro_fd139, %numbapro_fd137, %numbapro_fd138;
	mov.f64 	%numbapro_fd141, 0d3EC71DE8E6486D6B;
	fma.rn.f64 	%numbapro_fd142, %numbapro_fd140, %numbapro_fd137, %numbapro_fd141;
	mov.f64 	%numbapro_fd143, 0d3EFA019A6B2464C5;
	fma.rn.f64 	%numbapro_fd144, %numbapro_fd142, %numbapro_fd137, %numbapro_fd143;
	mov.f64 	%numbapro_fd145, 0d3F2A01A0171064A5;
	fma.rn.f64 	%numbapro_fd146, %numbapro_fd144, %numbapro_fd137, %numbapro_fd145;
	mov.f64 	%numbapro_fd147, 0d3F56C16C17F29C8D;
	fma.rn.f64 	%numbapro_fd148, %numbapro_fd146, %numbapro_fd137, %numbapro_fd147;
	mov.f64 	%numbapro_fd149, 0d3F8111111111A24E;
	fma.rn.f64 	%numbapro_fd150, %numbapro_fd148, %numbapro_fd137, %numbapro_fd149;
	mov.f64 	%numbapro_fd151, 0d3FA555555555211D;
	fma.rn.f64 	%numbapro_fd152, %numbapro_fd150, %numbapro_fd137, %numbapro_fd151;
	mov.f64 	%numbapro_fd153, 0d3FC5555555555530;
	fma.rn.f64 	%numbapro_fd154, %numbapro_fd152, %numbapro_fd137, %numbapro_fd153;
	mov.f64 	%numbapro_fd155, 0d3FE0000000000005;
	fma.rn.f64 	%numbapro_fd156, %numbapro_fd154, %numbapro_fd137, %numbapro_fd155;
	fma.rn.f64 	%numbapro_fd158, %numbapro_fd156, %numbapro_fd137, %numbapro_fd169;
	fma.rn.f64 	%numbapro_fd11, %numbapro_fd158, %numbapro_fd137, %numbapro_fd169;
	shl.b32 	%numbapro_r15, %numbapro_r14, 20;
	add.s32 	%numbapro_r16, %numbapro_r15, 1072693248;
	abs.s32 	%numbapro_r25, %numbapro_r14;
	setp.lt.s32 	%numbapro_p20, %numbapro_r25, 1021;
	@%numbapro_p20 bra 	BB1_17;

	add.s32 	%numbapro_r26, %numbapro_r15, 1130364928;
	setp.lt.s32 	%numbapro_p21, %numbapro_r14, 0;
	mov.u32 	%numbapro_r27, 0;
	selp.b32 	%numbapro_r28, %numbapro_r26, %numbapro_r16, %numbapro_p21;
	shr.s32 	%numbapro_r29, %numbapro_r14, 31;
	add.s32 	%numbapro_r30, %numbapro_r29, 1073741824;
	and.b32  	%numbapro_r31, %numbapro_r30, -57671680;
	add.s32 	%numbapro_r32, %numbapro_r28, -1048576;
	mov.b64 	%numbapro_fd159, {%numbapro_r27, %numbapro_r31};
	mul.f64 	%numbapro_fd160, %numbapro_fd11, %numbapro_fd159;
	mov.b64 	%numbapro_fd161, {%numbapro_r27, %numbapro_r32};
	mul.f64 	%numbapro_fd171, %numbapro_fd160, %numbapro_fd161;
	bra.uni 	BB1_18;

    BB1_17:
	mov.u32 	%numbapro_r33, 0;
	mov.b64 	%numbapro_fd162, {%numbapro_r33, %numbapro_r16};
	mul.f64 	%numbapro_fd171, %numbapro_fd162, %numbapro_fd11;

    BB1_18:
	abs.f64 	%numbapro_fd163, %numbapro_fd171;
	setp.eq.f64 	%numbapro_p22, %numbapro_fd163, %numbapro_fd2;
	@%numbapro_p22 bra 	BB1_20;

	fma.rn.f64 	%numbapro_fd171, %numbapro_fd171, %numbapro_fd9, %numbapro_fd171;

    BB1_20:
	mov.f64 	%numbapro_fd168, $1;
	setp.eq.f64 	%numbapro_p23, %numbapro_fd4, 0d3FF0000000000000;
	setp.lt.f64 	%numbapro_p24, %numbapro_fd168, 0d0000000000000000;
	and.pred  	%numbapro_p25, %numbapro_p24, %numbapro_p23;
	mov.b64 	%numbapro_rd2, %numbapro_fd171;
	xor.b64  	%numbapro_rd3, %numbapro_rd2, -9223372036854775808;
	mov.b64 	%numbapro_fd164, %numbapro_rd3;
	selp.f64 	%numbapro_fd173, %numbapro_fd164, %numbapro_fd171, %numbapro_p25;
	bra.uni 	BB1_33;

    BB1_21:
	mov.u64 	%numbapro_rd4, -2251799813685248;
	mov.b64 	%numbapro_fd173, %numbapro_rd4;
	bra.uni 	BB1_33;

    BB1_22:
	setp.lt.f64 	%numbapro_p26, %numbapro_fd31, 0d0000000000000000;
	@%numbapro_p26 bra 	BB1_24;

	neg.f64 	%numbapro_fd172, %numbapro_fd30;
	bra.uni 	BB1_25;

    BB1_24:
	mov.f64 	%numbapro_fd165, 0dBFF0000000000000;
	div.rn.f64 	%numbapro_fd172, %numbapro_fd165, %numbapro_fd30;

    BB1_25:
	mov.b64 	%numbapro_rd5, %numbapro_fd172;
	xor.b64  	%numbapro_rd6, %numbapro_rd5, -9223372036854775808;
	mov.b64 	%numbapro_fd166, %numbapro_rd6;
	setp.eq.f64 	%numbapro_p27, %numbapro_fd4, 0d3FF0000000000000;
	selp.f64 	%numbapro_fd173, %numbapro_fd166, %numbapro_fd172, %numbapro_p27;
	bra.uni 	BB1_33;

    BB1_26:
	setp.eq.f64 	%numbapro_p28, %numbapro_fd4, 0d3FF0000000000000;
	selp.f64 	%numbapro_fd173, %numbapro_fd30, 0d0000000000000000, %numbapro_p28;
	setp.geu.f64 	%numbapro_p29, %numbapro_fd31, 0d0000000000000000;
	@%numbapro_p29 bra 	BB1_33;

	rcp.rn.f64 	%numbapro_fd173, %numbapro_fd173;
	bra.uni 	BB1_33;

    BB1_28:
	setp.eq.f64 	%numbapro_p30, %numbapro_fd30, 0dBFF0000000000000;
	mov.f64 	%numbapro_fd173, 0d3FF0000000000000;
	@%numbapro_p30 bra 	BB1_33;

	setp.gt.f64 	%numbapro_p31, %numbapro_fd1, 0d3FF0000000000000;
	selp.f64 	%numbapro_fd173, %numbapro_fd2, 0d0000000000000000, %numbapro_p31;
	setp.geu.f64 	%numbapro_p32, %numbapro_fd31, 0d0000000000000000;
	@%numbapro_p32 bra 	BB1_33;

	rcp.rn.f64 	%numbapro_fd173, %numbapro_fd173;
	bra.uni 	BB1_33;

    BB1_31:
	{
	.reg .b32 %temp;
	mov.b64 	{%temp, %numbapro_r34}, %numbapro_fd31;
	}
	setp.lt.s32 	%numbapro_p33, %numbapro_r34, 0;
	selp.f64 	%numbapro_fd173, 0d0000000000000000, %numbapro_fd2, %numbapro_p33;
	bra.uni 	BB1_33;
    
    BB1_32:
	add.f64 	%numbapro_fd173, %numbapro_fd30, %numbapro_fd31;
    
    BB1_33:
	mov.f64	$0, %numbapro_fd173;
}
'''
_pow_f64_constrains = '=d,d,d'
pow_f64 = InlineAsm.get(_pow_f64_functype, _pow_f64_ptx, _pow_f64_constrains)
