.intel_syntax noprefix
mov    rax,QWORD PTR [rsp+0x18]
movzx  eax,WORD PTR [rax+r13*4]
lea    rsi,[rax+0x1]
mov    edx,esi
shl    edx,0xc
cmp    rdx,r12
shl    eax,0x5
shl    esi,0x5
cmp    rsi,r15
cmp    rsi,rbx
cmp    r13,0x40
shl    eax,0x2
mov    rcx,QWORD PTR [rsp+0x120]
vbroadcastss ymm1,DWORD PTR [rcx+rax*1]
vmulps ymm1,ymm0,ymm1
mov    rcx,QWORD PTR [rsp+0x28]
mov    eax,DWORD PTR [rcx+rax*1]
shl    eax,0x3
vmovd  xmm2,eax
vpbroadcastd ymm2,xmm2
vpsubd ymm2,ymm6,ymm2
vcvtdq2ps ymm2,ymm2
vmulps ymm1,ymm1,ymm2
mov    rax,r13
shl    rax,0x5
vaddps ymm1,ymm1,YMMWORD PTR [rsp+rax*1+0x960]
vmovaps YMMWORD PTR [rsp+rax*1+0x960],ymm1
