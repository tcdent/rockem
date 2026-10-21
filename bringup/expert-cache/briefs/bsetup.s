.intel_syntax noprefix
cmp    r13,r11
mov    r11,r15
mov    rax,QWORD PTR [rsp+0x18]
movzx  r15d,WORD PTR [rax+r13*4]
mov    eax,r15d
shl    eax,0xc
lea    rsi,[r15+0x1]
mov    edx,esi
shl    edx,0xc
cmp    rdx,r12
shl    esi,0x5
cmp    rsi,r11
cmp    rsi,rbx
lea    rcx,[r13+0x1]
cmp    rcx,r10
mov    rcx,QWORD PTR [rsp+0x18]
movzx  r15d,WORD PTR [rcx+r13*4+0x4]
mov    r8d,r15d
shl    r8d,0xc
lea    rsi,[r15+0x1]
mov    edx,esi
shl    edx,0xc
cmp    rdx,r12
shl    esi,0x5
cmp    rsi,r11
cmp    rsi,rbx
lea    rcx,[r13+0x2]
cmp    rcx,r10
mov    rcx,QWORD PTR [rsp+0x18]
movzx  r15d,WORD PTR [rcx+r13*4+0x8]
mov    edi,r15d
shl    edi,0xc
lea    rsi,[r15+0x1]
mov    edx,esi
shl    edx,0xc
cmp    rdx,r12
shl    esi,0x5
cmp    rsi,r11
cmp    rsi,rbx
lea    rcx,[r13+0x3]
cmp    rcx,r10
mov    rcx,QWORD PTR [rsp+0x18]
movzx  r15d,WORD PTR [rcx+r13*4+0xc]
mov    ecx,r15d
shl    ecx,0xc
lea    rsi,[r15+0x1]
mov    edx,esi
shl    edx,0xc
cmp    rdx,r12
shl    esi,0x5
cmp    rsi,r11
cmp    rsi,rbx
mov    rdx,QWORD PTR [rsp+0x20]
add    rax,rdx
add    r8,rdx
add    rdi,rdx
add    rcx,QWORD PTR [rsp+0x48]
vpxor  xmm3,xmm3,xmm3
xor    edx,edx
vpxor  xmm5,xmm5,xmm5
vpxor  xmm4,xmm4,xmm4
vpxor  xmm6,xmm6,xmm6
mov    r15,r11
mov    r11,r10
mov    r10,QWORD PTR [rsp+0x40]
data16 cs nop WORD PTR [rax+rax*1+0x0]
