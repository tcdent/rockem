.intel_syntax noprefix
vpbroadcastd ymm8,DWORD PTR [rax+rdx*1-0x1c]
vmovdqa ymm9,YMMWORD PTR [rsp+rdx*8+0x140]
vmovdqa ymm10,YMMWORD PTR [rsp+rdx*8+0x160]
vmovdqa ymm11,YMMWORD PTR [rsp+rdx*8+0x180]
vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x1a0]
vpmaddubsw ymm8,ymm9,ymm8
vpbroadcastd ymm12,DWORD PTR [r8+rdx*1-0x1c]
vpmaddubsw ymm12,ymm9,ymm12
vpbroadcastd ymm13,DWORD PTR [rdi+rdx*1-0x1c]
vpbroadcastd ymm14,DWORD PTR [rcx+rdx*1]
vpmaddubsw ymm13,ymm9,ymm13
vpmaddubsw ymm9,ymm9,ymm14
vpbroadcastd ymm14,DWORD PTR [rax+rdx*1-0x18]
vpmaddubsw ymm14,ymm10,ymm14
vpaddw ymm8,ymm14,ymm8
vpbroadcastd ymm14,DWORD PTR [r8+rdx*1-0x18]
vpmaddubsw ymm14,ymm10,ymm14
vpaddw ymm12,ymm14,ymm12
vpbroadcastd ymm14,DWORD PTR [rdi+rdx*1-0x18]
vpmaddubsw ymm14,ymm10,ymm14
vpaddw ymm13,ymm14,ymm13
vpbroadcastd ymm14,DWORD PTR [rcx+rdx*1+0x4]
vpmaddubsw ymm10,ymm10,ymm14
vpaddw ymm14,ymm10,ymm9
vpbroadcastd ymm9,DWORD PTR [rax+rdx*1-0x14]
vpmaddubsw ymm9,ymm11,ymm9
vpbroadcastd ymm10,DWORD PTR [r8+rdx*1-0x14]
vpmaddubsw ymm15,ymm11,ymm10
vpbroadcastd ymm10,DWORD PTR [rdi+rdx*1-0x14]
vpbroadcastd ymm1,DWORD PTR [rcx+rdx*1+0x8]
vpmaddubsw ymm2,ymm11,ymm10
vpmaddubsw ymm1,ymm11,ymm1
vpbroadcastd ymm10,DWORD PTR [rax+rdx*1-0x10]
vpmaddubsw ymm10,ymm7,ymm10
vpaddw ymm9,ymm10,ymm9
vpbroadcastd ymm11,DWORD PTR [r8+rdx*1-0x10]
vpaddw ymm10,ymm9,ymm8
vpmaddubsw ymm8,ymm7,ymm11
vpaddw ymm8,ymm8,ymm15
vpaddw ymm8,ymm8,ymm12
vpbroadcastd ymm9,DWORD PTR [rdi+rdx*1-0x10]
vpmaddubsw ymm9,ymm7,ymm9
vpaddw ymm2,ymm9,ymm2
vpaddw ymm9,ymm13,ymm2
vpbroadcastd ymm2,DWORD PTR [rcx+rdx*1+0xc]
vpmaddubsw ymm2,ymm7,ymm2
vpaddw ymm1,ymm2,ymm1
vpaddw ymm7,ymm14,ymm1
vpbroadcastd ymm1,DWORD PTR [rax+rdx*1-0xc]
vmovdqa ymm2,YMMWORD PTR [rsp+rdx*8+0x1c0]
vpmaddubsw ymm1,ymm2,ymm1
vpbroadcastd ymm11,DWORD PTR [r8+rdx*1-0xc]
vpmaddubsw ymm11,ymm2,ymm11
vpbroadcastd ymm12,DWORD PTR [rdi+rdx*1-0xc]
vpmaddubsw ymm12,ymm2,ymm12
vpbroadcastd ymm13,DWORD PTR [rcx+rdx*1+0x10]
vpmaddubsw ymm2,ymm2,ymm13
vmovdqa ymm13,YMMWORD PTR [rsp+rdx*8+0x1e0]
vpbroadcastd ymm14,DWORD PTR [rax+rdx*1-0x8]
vpmaddubsw ymm14,ymm13,ymm14
vpaddw ymm1,ymm14,ymm1
vpbroadcastd ymm14,DWORD PTR [r8+rdx*1-0x8]
vpmaddubsw ymm14,ymm13,ymm14
vpaddw ymm11,ymm14,ymm11
vpbroadcastd ymm14,DWORD PTR [rdi+rdx*1-0x8]
vpmaddubsw ymm14,ymm13,ymm14
vpaddw ymm12,ymm14,ymm12
vpbroadcastd ymm14,DWORD PTR [rcx+rdx*1+0x14]
vpmaddubsw ymm13,ymm13,ymm14
vpaddw ymm2,ymm13,ymm2
vmovdqa ymm13,YMMWORD PTR [rsp+rdx*8+0x200]
vpbroadcastd ymm14,DWORD PTR [rax+rdx*1-0x4]
vpmaddubsw ymm14,ymm13,ymm14
vpaddw ymm1,ymm14,ymm1
vpbroadcastw ymm14,WORD PTR [rip+0x6e205]        
vpaddw ymm1,ymm10,ymm1
vpbroadcastd ymm10,DWORD PTR [r8+rdx*1-0x4]
vpmaddubsw ymm10,ymm13,ymm10
vpaddw ymm10,ymm10,ymm11
vpbroadcastd ymm11,DWORD PTR [rdi+rdx*1-0x4]
vpaddw ymm8,ymm10,ymm8
vpmaddubsw ymm10,ymm13,ymm11
vpaddw ymm10,ymm10,ymm12
vpaddw ymm9,ymm10,ymm9
vpbroadcastd ymm10,DWORD PTR [rcx+rdx*1+0x18]
vpmaddubsw ymm10,ymm13,ymm10
vpaddw ymm2,ymm10,ymm2
vpaddw ymm2,ymm2,ymm7
vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x220]
vpbroadcastd ymm10,DWORD PTR [rax+rdx*1]
vpmaddubsw ymm10,ymm7,ymm10
vpaddw ymm1,ymm10,ymm1
vpbroadcastd ymm10,DWORD PTR [r8+rdx*1]
vpmaddubsw ymm10,ymm7,ymm10
vpaddw ymm8,ymm10,ymm8
vpbroadcastd ymm10,DWORD PTR [rdi+rdx*1]
vpmaddubsw ymm10,ymm7,ymm10
vpaddw ymm9,ymm10,ymm9
vpbroadcastd ymm10,DWORD PTR [rcx+rdx*1+0x1c]
vpmaddubsw ymm7,ymm7,ymm10
vpaddw ymm2,ymm7,ymm2
vpmaddwd ymm1,ymm14,ymm1
vpaddd ymm6,ymm1,ymm6
vpmaddwd ymm1,ymm8,ymm14
vpaddd ymm4,ymm1,ymm4
vpmaddwd ymm1,ymm9,ymm14
vpaddd ymm5,ymm1,ymm5
vpmaddwd ymm1,ymm14,ymm2
vpaddd ymm3,ymm1,ymm3
add    rdx,0x20
