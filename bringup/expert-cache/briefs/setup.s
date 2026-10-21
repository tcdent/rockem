.intel_syntax noprefix
mov    rcx,rax
shl    rcx,0x8
add    rcx,r14
mov    rsi,QWORD PTR [rsp+0xf8]
vmovdqu ymm4,YMMWORD PTR [rsi+rcx*1]
vmovdqu ymm1,YMMWORD PTR [rsi+rcx*1+0x40]
vmovdqu ymm6,YMMWORD PTR [rsi+rcx*1+0x80]
vmovdqu ymm3,YMMWORD PTR [rsi+rcx*1+0xc0]
mov    rdx,QWORD PTR [rsp+0x100]
vmovdqu ymm0,YMMWORD PTR [rdx+rcx*1]
vmovdqa YMMWORD PTR [rsp+0x80],ymm0
vmovdqu ymm5,YMMWORD PTR [rdx+rcx*1+0x40]
vmovdqu ymm15,YMMWORD PTR [rdx+rcx*1+0x80]
vmovdqa YMMWORD PTR [rsp+0x60],ymm15
vmovdqu ymm7,YMMWORD PTR [rdx+rcx*1+0xc0]
vmovdqa YMMWORD PTR [rsp+0x120],ymm7
vmovdqa ymm2,YMMWORD PTR [rsp+0x540]
vpand  ymm8,ymm4,ymm2
vpand  ymm9,ymm1,ymm2
vpunpckldq ymm10,ymm8,ymm9
vpunpckhdq ymm8,ymm8,ymm9
vpand  ymm9,ymm6,ymm2
vpand  ymm11,ymm3,ymm2
vpunpckldq ymm12,ymm9,ymm11
vpunpckhdq ymm9,ymm9,ymm11
vpand  ymm11,ymm0,ymm2
vpand  ymm13,ymm5,ymm2
vpunpckldq ymm14,ymm11,ymm13
vpunpckhdq ymm11,ymm11,ymm13
vpand  ymm13,ymm15,ymm2
vpand  ymm15,ymm7,ymm2
vpunpckldq ymm0,ymm13,ymm15
vpunpckhdq ymm13,ymm13,ymm15
vpunpcklqdq ymm15,ymm10,ymm12
vpunpckhqdq ymm10,ymm10,ymm12
vpunpcklqdq ymm12,ymm8,ymm9
vpunpckhqdq ymm8,ymm8,ymm9
vpunpcklqdq ymm9,ymm14,ymm0
vpunpckhqdq ymm0,ymm14,ymm0
vpunpcklqdq ymm14,ymm11,ymm13
vpunpckhqdq ymm11,ymm11,ymm13
vinserti128 ymm13,ymm15,xmm9,0x1
vmovdqa YMMWORD PTR [rsp+0x140],ymm13
vinserti128 ymm13,ymm10,xmm0,0x1
vmovdqa YMMWORD PTR [rsp+0x160],ymm13
vinserti128 ymm13,ymm12,xmm14,0x1
vmovdqa YMMWORD PTR [rsp+0x180],ymm13
vinserti128 ymm13,ymm8,xmm11,0x1
vmovdqa YMMWORD PTR [rsp+0x1a0],ymm13
vpsrlw ymm13,ymm4,0x4
vpsrlw ymm1,ymm1,0x4
vpsrlw ymm4,ymm6,0x4
vpsrlw ymm3,ymm3,0x4
vmovdqa ymm6,YMMWORD PTR [rsp+0x80]
vpsrlw ymm6,ymm6,0x4
vpsrlw ymm5,ymm5,0x4
vmovdqa ymm7,YMMWORD PTR [rsp+0x60]
vpsrlw ymm7,ymm7,0x4
vmovdqa YMMWORD PTR [rsp+0x80],ymm7
vmovdqa ymm7,YMMWORD PTR [rsp+0x120]
vpsrlw ymm7,ymm7,0x4
vperm2i128 ymm9,ymm15,ymm9,0x31
vmovdqa YMMWORD PTR [rsp+0x1c0],ymm9
vperm2i128 ymm0,ymm10,ymm0,0x31
vmovdqa YMMWORD PTR [rsp+0x1e0],ymm0
vperm2i128 ymm0,ymm12,ymm14,0x31
vmovdqa YMMWORD PTR [rsp+0x200],ymm0
vperm2i128 ymm0,ymm8,ymm11,0x31
vmovdqa YMMWORD PTR [rsp+0x220],ymm0
vpbroadcastb ymm10,BYTE PTR [rip+0x6eb48]        
vpand  ymm0,ymm13,ymm10
vpand  ymm1,ymm10,ymm1
vpunpckldq ymm8,ymm0,ymm1
vpunpckhdq ymm0,ymm0,ymm1
vpand  ymm1,ymm10,ymm4
vpand  ymm4,ymm10,ymm3
vpunpckldq ymm3,ymm1,ymm4
vpunpckhdq ymm1,ymm1,ymm4
vpand  ymm6,ymm10,ymm6
vpand  ymm4,ymm10,ymm5
vpunpckldq ymm5,ymm6,ymm4
vpunpckhdq ymm9,ymm6,ymm4
vpand  ymm4,ymm10,YMMWORD PTR [rsp+0x80]
vpand  ymm6,ymm10,ymm7
vpunpckldq ymm7,ymm4,ymm6
vpunpckhdq ymm4,ymm4,ymm6
vpunpcklqdq ymm6,ymm8,ymm3
vpunpckhqdq ymm3,ymm8,ymm3
vpunpcklqdq ymm8,ymm0,ymm1
vpunpckhqdq ymm0,ymm0,ymm1
vpunpcklqdq ymm1,ymm5,ymm7
vpunpckhqdq ymm5,ymm5,ymm7
vpunpcklqdq ymm7,ymm9,ymm4
vpunpckhqdq ymm9,ymm9,ymm4
vinserti128 ymm4,ymm6,xmm1,0x1
vmovdqa YMMWORD PTR [rsp+0x240],ymm4
vinserti128 ymm4,ymm3,xmm5,0x1
vmovdqa YMMWORD PTR [rsp+0x260],ymm4
vinserti128 ymm4,ymm8,xmm7,0x1
vmovdqa YMMWORD PTR [rsp+0x280],ymm4
vinserti128 ymm4,ymm0,xmm9,0x1
vmovdqa YMMWORD PTR [rsp+0x2a0],ymm4
vperm2i128 ymm1,ymm6,ymm1,0x31
vmovdqa YMMWORD PTR [rsp+0x2c0],ymm1
vperm2i128 ymm1,ymm3,ymm5,0x31
vmovdqa YMMWORD PTR [rsp+0x2e0],ymm1
vperm2i128 ymm1,ymm8,ymm7,0x31
vmovdqa YMMWORD PTR [rsp+0x300],ymm1
vperm2i128 ymm0,ymm0,ymm9,0x31
vmovdqa YMMWORD PTR [rsp+0x320],ymm0
vmovdqu ymm0,YMMWORD PTR [rsi+rcx*1+0x20]
vmovdqa YMMWORD PTR [rsp+0x80],ymm0
vmovdqu ymm1,YMMWORD PTR [rsi+rcx*1+0x60]
vmovdqu ymm11,YMMWORD PTR [rsi+rcx*1+0xa0]
vmovdqa YMMWORD PTR [rsp+0x60],ymm11
vmovdqu ymm3,YMMWORD PTR [rsi+rcx*1+0xe0]
vmovdqu ymm4,YMMWORD PTR [rdx+rcx*1+0x20]
vmovdqu ymm5,YMMWORD PTR [rdx+rcx*1+0x60]
vmovdqu ymm6,YMMWORD PTR [rdx+rcx*1+0xa0]
vmovdqu ymm7,YMMWORD PTR [rdx+rcx*1+0xe0]
vpand  ymm8,ymm0,ymm2
vpand  ymm9,ymm1,ymm2
vpunpckldq ymm10,ymm8,ymm9
vpunpckhdq ymm8,ymm8,ymm9
vpand  ymm9,ymm11,ymm2
vpand  ymm11,ymm3,ymm2
vpunpckldq ymm12,ymm9,ymm11
vpunpckhdq ymm9,ymm9,ymm11
vpand  ymm11,ymm4,ymm2
vpand  ymm13,ymm5,ymm2
vpunpckldq ymm14,ymm11,ymm13
vpunpckhdq ymm11,ymm11,ymm13
vpand  ymm13,ymm6,ymm2
vpand  ymm15,ymm7,ymm2
vpunpckldq ymm0,ymm13,ymm15
vpunpckhdq ymm13,ymm13,ymm15
vpunpcklqdq ymm15,ymm10,ymm12
vpunpckhqdq ymm10,ymm10,ymm12
vpunpcklqdq ymm12,ymm8,ymm9
vpunpckhqdq ymm8,ymm8,ymm9
vpunpcklqdq ymm9,ymm14,ymm0
vpunpckhqdq ymm0,ymm14,ymm0
vpunpcklqdq ymm14,ymm11,ymm13
vpunpckhqdq ymm11,ymm11,ymm13
vinserti128 ymm13,ymm15,xmm9,0x1
vmovdqa YMMWORD PTR [rsp+0x340],ymm13
vinserti128 ymm13,ymm10,xmm0,0x1
vmovdqa YMMWORD PTR [rsp+0x360],ymm13
vinserti128 ymm13,ymm12,xmm14,0x1
vmovdqa YMMWORD PTR [rsp+0x380],ymm13
vinserti128 ymm13,ymm8,xmm11,0x1
vmovdqa YMMWORD PTR [rsp+0x3a0],ymm13
vperm2i128 ymm9,ymm15,ymm9,0x31
vmovdqa YMMWORD PTR [rsp+0x3c0],ymm9
vperm2i128 ymm0,ymm10,ymm0,0x31
vmovdqa YMMWORD PTR [rsp+0x3e0],ymm0
vperm2i128 ymm0,ymm12,ymm14,0x31
vmovdqa YMMWORD PTR [rsp+0x400],ymm0
vperm2i128 ymm0,ymm8,ymm11,0x31
vmovdqa ymm2,YMMWORD PTR [rsp+0x80]
vpsrlw ymm8,ymm2,0x4
vpsrlw ymm1,ymm1,0x4
vmovdqa ymm2,YMMWORD PTR [rsp+0x60]
vpsrlw ymm2,ymm2,0x4
vpsrlw ymm3,ymm3,0x4
vpsrlw ymm4,ymm4,0x4
vpsrlw ymm5,ymm5,0x4
vpsrlw ymm6,ymm6,0x4
vpsrlw ymm7,ymm7,0x4
vmovdqa YMMWORD PTR [rsp+0x420],ymm0
vpbroadcastb ymm9,BYTE PTR [rip+0x6e8f9]        
vpand  ymm0,ymm8,ymm9
vpand  ymm1,ymm9,ymm1
vpunpckldq ymm8,ymm0,ymm1
vpunpckhdq ymm0,ymm0,ymm1
vpand  ymm1,ymm9,ymm2
vpand  ymm2,ymm9,ymm3
vpunpckldq ymm3,ymm1,ymm2
vpunpckhdq ymm1,ymm1,ymm2
vpand  ymm2,ymm9,ymm4
vpand  ymm4,ymm9,ymm5
vpunpckldq ymm5,ymm2,ymm4
vpunpckhdq ymm2,ymm2,ymm4
vpand  ymm4,ymm9,ymm6
vpand  ymm6,ymm9,ymm7
vpunpckldq ymm7,ymm4,ymm6
vpunpckhdq ymm4,ymm4,ymm6
vpunpcklqdq ymm6,ymm8,ymm3
vpunpckhqdq ymm3,ymm8,ymm3
vpunpcklqdq ymm8,ymm0,ymm1
vpunpckhqdq ymm0,ymm0,ymm1
vpunpcklqdq ymm1,ymm5,ymm7
vpunpckhqdq ymm5,ymm5,ymm7
vpunpcklqdq ymm7,ymm2,ymm4
vpunpckhqdq ymm2,ymm2,ymm4
vinserti128 ymm4,ymm6,xmm1,0x1
vmovdqa YMMWORD PTR [rsp+0x440],ymm4
vinserti128 ymm4,ymm3,xmm5,0x1
vmovdqa YMMWORD PTR [rsp+0x460],ymm4
vinserti128 ymm4,ymm8,xmm7,0x1
vmovdqa YMMWORD PTR [rsp+0x480],ymm4
vinserti128 ymm4,ymm0,xmm2,0x1
vmovdqa YMMWORD PTR [rsp+0x4a0],ymm4
vperm2i128 ymm1,ymm6,ymm1,0x31
vmovdqa YMMWORD PTR [rsp+0x4c0],ymm1
vperm2i128 ymm1,ymm3,ymm5,0x31
vmovdqa YMMWORD PTR [rsp+0x4e0],ymm1
vperm2i128 ymm1,ymm8,ymm7,0x31
vmovdqa YMMWORD PTR [rsp+0x500],ymm1
vperm2i128 ymm0,ymm0,ymm2,0x31
vmovdqa YMMWORD PTR [rsp+0x520],ymm0
lea    rcx,[rax+0x1]
test   r10,r10
mov    QWORD PTR [rsp+0x108],rcx
mov    rcx,rax
shl    rcx,0x4
mov    rdx,QWORD PTR [rsp+0xf0]
vmovups xmm0,XMMWORD PTR [rdx+rcx*1]
mov    rdx,QWORD PTR [rsp+0xe8]
vinsertf128 ymm0,ymm0,XMMWORD PTR [rdx+rcx*1],0x1
mov    r9,rax
shl    r9,0x7
add    r9,QWORD PTR [rsp+0x50]
mov    rcx,QWORD PTR [rsp+0xe0]
lea    rcx,[rcx+rax*4]
mov    QWORD PTR [rsp+0x120],rcx
mov    rcx,QWORD PTR [rsp+0xd8]
lea    rax,[rcx+rax*4]
mov    QWORD PTR [rsp+0x28],rax
vmovaps ymm1,YMMWORD PTR [rsp+0x140]
vmovaps YMMWORD PTR [rsp+0x940],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x160]
vmovaps YMMWORD PTR [rsp+0x920],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x180]
vmovaps YMMWORD PTR [rsp+0x900],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x1a0]
vmovaps YMMWORD PTR [rsp+0x8e0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x1c0]
vmovaps YMMWORD PTR [rsp+0x8c0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x1e0]
vmovaps YMMWORD PTR [rsp+0x8a0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x200]
vmovaps YMMWORD PTR [rsp+0x880],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x220]
vmovaps YMMWORD PTR [rsp+0x860],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x240]
vmovaps YMMWORD PTR [rsp+0x840],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x260]
vmovaps YMMWORD PTR [rsp+0x820],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x280]
vmovaps YMMWORD PTR [rsp+0x800],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x2a0]
vmovaps YMMWORD PTR [rsp+0x7e0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x2c0]
vmovaps YMMWORD PTR [rsp+0x7c0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x2e0]
vmovaps YMMWORD PTR [rsp+0x7a0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x300]
vmovaps YMMWORD PTR [rsp+0x780],ymm1
xor    r13d,r13d
vmovaps ymm1,YMMWORD PTR [rsp+0x320]
vmovaps YMMWORD PTR [rsp+0x760],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x340]
vmovaps YMMWORD PTR [rsp+0x740],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x360]
vmovaps YMMWORD PTR [rsp+0x720],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x380]
vmovaps YMMWORD PTR [rsp+0x700],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x3a0]
vmovaps YMMWORD PTR [rsp+0x6e0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x3c0]
vmovaps YMMWORD PTR [rsp+0x6c0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x3e0]
vmovaps YMMWORD PTR [rsp+0x6a0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x400]
vmovaps YMMWORD PTR [rsp+0x680],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x420]
vmovaps YMMWORD PTR [rsp+0x660],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x440]
vmovaps YMMWORD PTR [rsp+0x640],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x460]
vmovaps YMMWORD PTR [rsp+0x620],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x480]
vmovaps YMMWORD PTR [rsp+0x600],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x4a0]
vmovaps YMMWORD PTR [rsp+0x5e0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x4c0]
vmovaps YMMWORD PTR [rsp+0x5c0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x4e0]
vmovaps YMMWORD PTR [rsp+0x5a0],ymm1
vmovaps ymm1,YMMWORD PTR [rsp+0x500]
vmovaps YMMWORD PTR [rsp+0x580],ymm1
vmovdqa ymm1,YMMWORD PTR [rsp+0x520]
vmovdqa YMMWORD PTR [rsp+0x560],ymm1
