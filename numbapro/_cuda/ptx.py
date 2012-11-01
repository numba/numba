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


