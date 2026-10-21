
target-elf/x86_64-unknown-linux-musl/release/node:     file format elf64-x86-64


Disassembly of section .text:

00000000000d50f0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_>:
   d50f0:	push   rbp
   d50f1:	mov    rbp,rsp
   d50f4:	push   r15
   d50f6:	push   r14
   d50f8:	push   r13
   d50fa:	push   r12
   d50fc:	push   rbx
   d50fd:	and    rsp,0xffffffffffffffe0
   d5101:	sub    rsp,0xfe0
   d5108:	mov    QWORD PTR [rsp],0x0
   d5110:	sub    rsp,0x1a0
   d5117:	mov    QWORD PTR [rsp+0x110],0x20
   d5123:	mov    QWORD PTR [rsp+0x118],rdx
   d512b:	cmp    rdx,0x41
   d512f:	jae    d6e3c <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d4c>
   d5135:	mov    r13,r8
   d5138:	mov    r15,rcx
   d513b:	mov    rax,rdx
   d513e:	mov    r11,rsi
   d5141:	mov    r14,rdi
   d5144:	cmp    rdx,0x1
   d5148:	jne    d53f9 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x309>
   d514e:	cmp    QWORD PTR [r15+0x8],0x0
   d5153:	je     d6f1a <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1e2a>
   d5159:	mov    rax,QWORD PTR [r15]
   d515c:	movzx  edi,WORD PTR [rax]
   d515f:	mov    rdx,QWORD PTR [r15+0x18]
   d5163:	mov    eax,edi
   d5165:	shl    eax,0xc
   d5168:	lea    rsi,[rdi+0x1]
   d516c:	mov    r8d,esi
   d516f:	shl    r8d,0xc
   d5173:	cmp    r8,rdx
   d5176:	ja     d6e8e <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d9e>
   d517c:	mov    rdx,QWORD PTR [r15+0x28]
   d5180:	shl    edi,0x5
   d5183:	shl    esi,0x5
   d5186:	cmp    rsi,rdx
   d5189:	ja     d6ea0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1db0>
   d518f:	mov    rdx,QWORD PTR [r15+0x38]
   d5193:	cmp    rsi,rdx
   d5196:	ja     d6eac <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dbc>
   d519c:	mov    rsi,QWORD PTR [r15+0x10]
   d51a0:	shl    edi,0x2
   d51a3:	mov    rcx,QWORD PTR [r15+0x20]
   d51a7:	add    rcx,rdi
   d51aa:	add    rdi,QWORD PTR [r15+0x30]
   d51ae:	mov    rdx,QWORD PTR [r13+0x0]
   d51b2:	add    r14,0xe0
   d51b9:	add    rax,rsi
   d51bc:	add    rax,0x60
   d51c0:	xor    esi,esi
   d51c2:	vpbroadcastb ymm0,BYTE PTR [rip+0x6f0e9]        # 1442b4 <_fini+0x1b94>
   d51cb:	vpbroadcastw ymm1,WORD PTR [rip+0x6f0e2]        # 1442b6 <_fini+0x1b96>
   d51d4:	data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
   d51e0:	vxorps xmm2,xmm2,xmm2
   d51e4:	xor    r8d,r8d
   d51e7:	xor    r9d,r9d
   d51ea:	nop    WORD PTR [rax+rax*1+0x0]
   d51f0:	vmovdqu ymm6,YMMWORD PTR [rax+r8*1-0x60]
   d51f7:	vmovdqu ymm5,YMMWORD PTR [rax+r8*1-0x40]
   d51fe:	vmovdqu ymm4,YMMWORD PTR [rax+r8*1-0x20]
   d5205:	vmovdqu ymm3,YMMWORD PTR [rax+r8*1]
   d520b:	vmovdqu ymm7,YMMWORD PTR [r14+r8*2-0xe0]
   d5215:	vmovdqu ymm8,YMMWORD PTR [r14+r8*2-0xc0]
   d521f:	vmovdqu ymm9,YMMWORD PTR [r14+r8*2-0xa0]
   d5229:	vmovdqu ymm10,YMMWORD PTR [r14+r8*2-0x80]
   d5230:	vpsrlw ymm11,ymm7,0x4
   d5235:	vpand  ymm7,ymm7,ymm0
   d5239:	vpmaddubsw ymm7,ymm7,ymm6
   d523e:	vpand  ymm11,ymm11,ymm0
   d5242:	vpmaddubsw ymm11,ymm11,ymm5
   d5247:	vpaddw ymm7,ymm11,ymm7
   d524b:	vpmaddwd ymm7,ymm7,ymm1
   d524f:	vpsrlw ymm11,ymm8,0x4
   d5255:	vpand  ymm8,ymm8,ymm0
   d5259:	vpmaddubsw ymm8,ymm8,ymm4
   d525e:	vpand  ymm11,ymm11,ymm0
   d5262:	vpmaddubsw ymm11,ymm11,ymm3
   d5267:	vpaddw ymm8,ymm11,ymm8
   d526c:	vpmaddwd ymm8,ymm8,ymm1
   d5270:	vpaddd ymm7,ymm8,ymm7
   d5274:	vpsrlw ymm8,ymm9,0x4
   d527a:	vpand  ymm9,ymm9,ymm0
   d527e:	vpmaddubsw ymm9,ymm9,ymm6
   d5283:	vpand  ymm8,ymm8,ymm0
   d5287:	vpmaddubsw ymm8,ymm8,ymm5
   d528c:	vpaddw ymm8,ymm8,ymm9
   d5291:	vpmaddwd ymm8,ymm8,ymm1
   d5295:	vpsrlw ymm9,ymm10,0x4
   d529b:	vpand  ymm10,ymm10,ymm0
   d529f:	vpmaddubsw ymm10,ymm10,ymm4
   d52a4:	vpand  ymm9,ymm9,ymm0
   d52a8:	vpmaddubsw ymm9,ymm9,ymm3
   d52ad:	vpaddw ymm9,ymm9,ymm10
   d52b2:	vpmaddwd ymm9,ymm9,ymm1
   d52b6:	vpaddd ymm8,ymm9,ymm8
   d52bb:	vphaddd ymm7,ymm7,ymm8
   d52c0:	vmovdqu ymm8,YMMWORD PTR [r14+r8*2-0x60]
   d52c7:	vpsrlw ymm9,ymm8,0x4
   d52cd:	vpand  ymm8,ymm8,ymm0
   d52d1:	vpmaddubsw ymm8,ymm8,ymm6
   d52d6:	vpand  ymm9,ymm9,ymm0
   d52da:	vpmaddubsw ymm9,ymm9,ymm5
   d52df:	vpaddw ymm8,ymm9,ymm8
   d52e4:	vpmaddwd ymm8,ymm8,ymm1
   d52e8:	vmovdqu ymm9,YMMWORD PTR [r14+r8*2-0x40]
   d52ef:	vpsrlw ymm10,ymm9,0x4
   d52f5:	vpand  ymm9,ymm9,ymm0
   d52f9:	vpmaddubsw ymm9,ymm9,ymm4
   d52fe:	vpand  ymm10,ymm10,ymm0
   d5302:	vpmaddubsw ymm10,ymm10,ymm3
   d5307:	vpaddw ymm9,ymm10,ymm9
   d530c:	vpmaddwd ymm9,ymm9,ymm1
   d5310:	vpaddd ymm8,ymm9,ymm8
   d5315:	vmovdqu ymm9,YMMWORD PTR [r14+r8*2-0x20]
   d531c:	vpsrlw ymm10,ymm9,0x4
   d5322:	vpand  ymm9,ymm9,ymm0
   d5326:	vpmaddubsw ymm6,ymm9,ymm6
   d532b:	vpand  ymm9,ymm10,ymm0
   d532f:	vpmaddubsw ymm5,ymm9,ymm5
   d5334:	vpaddw ymm5,ymm5,ymm6
   d5338:	vpmaddwd ymm5,ymm5,ymm1
   d533c:	vmovdqu ymm6,YMMWORD PTR [r14+r8*2]
   d5342:	vpsrlw ymm9,ymm6,0x4
   d5347:	vpand  ymm6,ymm6,ymm0
   d534b:	vpmaddubsw ymm4,ymm6,ymm4
   d5350:	vpand  ymm6,ymm9,ymm0
   d5354:	vpmaddubsw ymm3,ymm6,ymm3
   d5359:	vpaddw ymm3,ymm3,ymm4
   d535d:	vpmaddwd ymm3,ymm3,ymm1
   d5361:	vpaddd ymm3,ymm3,ymm5
   d5365:	vphaddd ymm3,ymm8,ymm3
   d536a:	vphaddd ymm3,ymm7,ymm3
   d536f:	vextracti128 xmm4,ymm3,0x1
   d5375:	mov    r10d,DWORD PTR [rdi+r9*1]
   d5379:	shl    r10d,0x3
   d537d:	neg    r10d
   d5380:	vmovd  xmm5,r10d
   d5385:	vpbroadcastd xmm5,xmm5
   d538a:	vpaddd xmm3,xmm5,xmm3
   d538e:	vpaddd xmm3,xmm3,xmm4
   d5392:	vcvtdq2ps xmm3,xmm3
   d5396:	vbroadcastss xmm4,DWORD PTR [rcx+r9*1]
   d539c:	vmulps xmm4,xmm4,XMMWORD PTR [r11+r9*4]
   d53a2:	vmulps xmm3,xmm4,xmm3
   d53a6:	vaddps xmm2,xmm2,xmm3
   d53aa:	add    r9,0x4
   d53ae:	sub    r8,0xffffffffffffff80
   d53b2:	cmp    r9,0x80
   d53b9:	jne    d51f0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x100>
   d53bf:	lea    r8,[rsi+0x1]
   d53c3:	shl    rsi,0x4
   d53c7:	vmovups XMMWORD PTR [rdx+rsi*1],xmm2
   d53cc:	add    r11,0x200
   d53d3:	add    r14,0x2000
   d53da:	mov    rsi,r8
   d53dd:	cmp    r8,0x8
   d53e1:	jne    d51e0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xf0>
   d53e7:	lea    rsp,[rbp-0x28]
   d53eb:	pop    rbx
   d53ec:	pop    r12
   d53ee:	pop    r13
   d53f0:	pop    r14
   d53f2:	pop    r15
   d53f4:	pop    rbp
   d53f5:	vzeroupper
   d53f8:	ret
   d53f9:	mov    QWORD PTR [rsp+0xc0],r11
   d5401:	lea    rdi,[rsp+0x140]
   d5409:	mov    QWORD PTR [rsp+0x38],0x0
   d5412:	lea    rbx,[rip+0x67299]        # 13c6b2 <memset>
   d5419:	mov    edx,0x400
   d541e:	xor    esi,esi
   d5420:	mov    QWORD PTR [rsp+0x40],rax
   d5425:	call   rbx
   d5427:	lea    rdi,[rsp+0x960]
   d542f:	mov    edx,0x800
   d5434:	xor    esi,esi
   d5436:	call   rbx
   d5438:	mov    r10,QWORD PTR [rsp+0x40]
   d543d:	mov    rax,QWORD PTR [r15]
   d5440:	mov    QWORD PTR [rsp+0x18],rax
   d5445:	mov    r11,QWORD PTR [r15+0x8]
   d5449:	mov    rax,QWORD PTR [r15+0x10]
   d544d:	mov    r12,QWORD PTR [r15+0x18]
   d5451:	mov    rdx,r15
   d5454:	mov    r15,QWORD PTR [r15+0x28]
   d5458:	mov    rbx,QWORD PTR [rdx+0x38]
   d545c:	mov    rcx,QWORD PTR [rdx+0x20]
   d5460:	mov    QWORD PTR [rsp+0xe0],rcx
   d5468:	mov    rcx,QWORD PTR [rdx+0x30]
   d546c:	mov    QWORD PTR [rsp+0xd8],rcx
   d5474:	mov    rcx,QWORD PTR [r13+0x0]
   d5478:	mov    rdx,r10
   d547b:	shl    rdx,0x5
   d547f:	mov    QWORD PTR [rsp+0xa8],rdx
   d5487:	mov    edx,r10d
   d548a:	and    edx,0x3
   d548d:	mov    esi,r10d
   d5490:	and    esi,0x7c
   d5493:	mov    QWORD PTR [rsp+0xd0],rsi
   d549b:	mov    QWORD PTR [rsp+0xa0],rdx
   d54a3:	shl    edx,0x5
   d54a6:	mov    QWORD PTR [rsp+0xc8],rdx
   d54ae:	vbroadcastss ymm0,DWORD PTR [rip+0x6e639]        # 143af0 <_fini+0x13d0>
   d54b7:	vmovaps YMMWORD PTR [rsp+0x540],ymm0
   d54c0:	lea    rdx,[rax+0x1c]
   d54c4:	mov    QWORD PTR [rsp+0xb8],rdx
   d54cc:	mov    QWORD PTR [rsp+0x50],rax
   d54d1:	add    rax,0x10
   d54d5:	mov    QWORD PTR [rsp+0xb0],rax
   d54dd:	mov    QWORD PTR [rsp+0x30],rcx
   d54e2:	test   r10,r10
   d54e5:	je     d550d <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x41d>
   d54e7:	lea    rdi,[rsp+0x960]
   d54ef:	xor    esi,esi
   d54f1:	mov    rdx,QWORD PTR [rsp+0xa8]
   d54f9:	mov    r13,r11
   d54fc:	vzeroupper
   d54ff:	call   QWORD PTR [rip+0xa5983]        # 17ae88 <_GLOBAL_OFFSET_TABLE_+0x288>
   d5505:	mov    r10,QWORD PTR [rsp+0x40]
   d550a:	mov    r11,r13
   d550d:	mov    rcx,QWORD PTR [rsp+0x38]
   d5512:	mov    rdx,rcx
   d5515:	or     rdx,0x1
   d5519:	mov    rsi,rcx
   d551c:	shl    rsi,0x9
   d5520:	mov    rax,QWORD PTR [rsp+0xc0]
   d5528:	add    rsi,rax
   d552b:	mov    QWORD PTR [rsp+0xf0],rsi
   d5533:	mov    rsi,rdx
   d5536:	shl    rsi,0x9
   d553a:	add    rsi,rax
   d553d:	mov    QWORD PTR [rsp+0xe8],rsi
   d5545:	shl    rcx,0xd
   d5549:	mov    QWORD PTR [rsp+0xf8],rcx
   d5551:	shl    rdx,0xd
   d5555:	mov    QWORD PTR [rsp+0x100],rdx
   d555d:	mov    rax,QWORD PTR [rsp+0x50]
   d5562:	mov    QWORD PTR [rsp+0x48],rax
   d5567:	mov    rax,QWORD PTR [rsp+0xb0]
   d556f:	mov    QWORD PTR [rsp+0x58],rax
   d5574:	mov    rax,QWORD PTR [rsp+0xb8]
   d557c:	mov    QWORD PTR [rsp+0x20],rax
   d5581:	xor    eax,eax
   d5583:	jmp    d55bf <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x4cf>
   d5585:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   d5590:	mov    rcx,QWORD PTR [rsp+0x108]
   d5598:	cmp    rcx,0x20
   d559c:	mov    rsi,QWORD PTR [rsp+0xd0]
   d55a4:	je     d6c9f <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1baf>
   d55aa:	sub    QWORD PTR [rsp+0x20],0xffffffffffffff80
   d55b0:	sub    QWORD PTR [rsp+0x58],0xffffffffffffff80
   d55b6:	sub    QWORD PTR [rsp+0x48],0xffffffffffffff80
   d55bc:	mov    rax,rcx
   d55bf:	mov    rcx,rax
   d55c2:	shl    rcx,0x8
   d55c6:	add    rcx,r14
   d55c9:	mov    rsi,QWORD PTR [rsp+0xf8]
   d55d1:	vmovdqu ymm4,YMMWORD PTR [rsi+rcx*1]
   d55d6:	vmovdqu ymm1,YMMWORD PTR [rsi+rcx*1+0x40]
   d55dc:	vmovdqu ymm6,YMMWORD PTR [rsi+rcx*1+0x80]
   d55e5:	vmovdqu ymm3,YMMWORD PTR [rsi+rcx*1+0xc0]
   d55ee:	mov    rdx,QWORD PTR [rsp+0x100]
   d55f6:	vmovdqu ymm0,YMMWORD PTR [rdx+rcx*1]
   d55fb:	vmovdqa YMMWORD PTR [rsp+0x80],ymm0
   d5604:	vmovdqu ymm5,YMMWORD PTR [rdx+rcx*1+0x40]
   d560a:	vmovdqu ymm15,YMMWORD PTR [rdx+rcx*1+0x80]
   d5613:	vmovdqa YMMWORD PTR [rsp+0x60],ymm15
   d5619:	vmovdqu ymm7,YMMWORD PTR [rdx+rcx*1+0xc0]
   d5622:	vmovdqa YMMWORD PTR [rsp+0x120],ymm7
   d562b:	vmovdqa ymm2,YMMWORD PTR [rsp+0x540]
   d5634:	vpand  ymm8,ymm4,ymm2
   d5638:	vpand  ymm9,ymm1,ymm2
   d563c:	vpunpckldq ymm10,ymm8,ymm9
   d5641:	vpunpckhdq ymm8,ymm8,ymm9
   d5646:	vpand  ymm9,ymm6,ymm2
   d564a:	vpand  ymm11,ymm3,ymm2
   d564e:	vpunpckldq ymm12,ymm9,ymm11
   d5653:	vpunpckhdq ymm9,ymm9,ymm11
   d5658:	vpand  ymm11,ymm0,ymm2
   d565c:	vpand  ymm13,ymm5,ymm2
   d5660:	vpunpckldq ymm14,ymm11,ymm13
   d5665:	vpunpckhdq ymm11,ymm11,ymm13
   d566a:	vpand  ymm13,ymm15,ymm2
   d566e:	vpand  ymm15,ymm7,ymm2
   d5672:	vpunpckldq ymm0,ymm13,ymm15
   d5677:	vpunpckhdq ymm13,ymm13,ymm15
   d567c:	vpunpcklqdq ymm15,ymm10,ymm12
   d5681:	vpunpckhqdq ymm10,ymm10,ymm12
   d5686:	vpunpcklqdq ymm12,ymm8,ymm9
   d568b:	vpunpckhqdq ymm8,ymm8,ymm9
   d5690:	vpunpcklqdq ymm9,ymm14,ymm0
   d5694:	vpunpckhqdq ymm0,ymm14,ymm0
   d5698:	vpunpcklqdq ymm14,ymm11,ymm13
   d569d:	vpunpckhqdq ymm11,ymm11,ymm13
   d56a2:	vinserti128 ymm13,ymm15,xmm9,0x1
   d56a8:	vmovdqa YMMWORD PTR [rsp+0x140],ymm13
   d56b1:	vinserti128 ymm13,ymm10,xmm0,0x1
   d56b7:	vmovdqa YMMWORD PTR [rsp+0x160],ymm13
   d56c0:	vinserti128 ymm13,ymm12,xmm14,0x1
   d56c6:	vmovdqa YMMWORD PTR [rsp+0x180],ymm13
   d56cf:	vinserti128 ymm13,ymm8,xmm11,0x1
   d56d5:	vmovdqa YMMWORD PTR [rsp+0x1a0],ymm13
   d56de:	vpsrlw ymm13,ymm4,0x4
   d56e3:	vpsrlw ymm1,ymm1,0x4
   d56e8:	vpsrlw ymm4,ymm6,0x4
   d56ed:	vpsrlw ymm3,ymm3,0x4
   d56f2:	vmovdqa ymm6,YMMWORD PTR [rsp+0x80]
   d56fb:	vpsrlw ymm6,ymm6,0x4
   d5700:	vpsrlw ymm5,ymm5,0x4
   d5705:	vmovdqa ymm7,YMMWORD PTR [rsp+0x60]
   d570b:	vpsrlw ymm7,ymm7,0x4
   d5710:	vmovdqa YMMWORD PTR [rsp+0x80],ymm7
   d5719:	vmovdqa ymm7,YMMWORD PTR [rsp+0x120]
   d5722:	vpsrlw ymm7,ymm7,0x4
   d5727:	vperm2i128 ymm9,ymm15,ymm9,0x31
   d572d:	vmovdqa YMMWORD PTR [rsp+0x1c0],ymm9
   d5736:	vperm2i128 ymm0,ymm10,ymm0,0x31
   d573c:	vmovdqa YMMWORD PTR [rsp+0x1e0],ymm0
   d5745:	vperm2i128 ymm0,ymm12,ymm14,0x31
   d574b:	vmovdqa YMMWORD PTR [rsp+0x200],ymm0
   d5754:	vperm2i128 ymm0,ymm8,ymm11,0x31
   d575a:	vmovdqa YMMWORD PTR [rsp+0x220],ymm0
   d5763:	vpbroadcastb ymm10,BYTE PTR [rip+0x6eb48]        # 1442b4 <_fini+0x1b94>
   d576c:	vpand  ymm0,ymm13,ymm10
   d5771:	vpand  ymm1,ymm10,ymm1
   d5775:	vpunpckldq ymm8,ymm0,ymm1
   d5779:	vpunpckhdq ymm0,ymm0,ymm1
   d577d:	vpand  ymm1,ymm10,ymm4
   d5781:	vpand  ymm4,ymm10,ymm3
   d5785:	vpunpckldq ymm3,ymm1,ymm4
   d5789:	vpunpckhdq ymm1,ymm1,ymm4
   d578d:	vpand  ymm6,ymm10,ymm6
   d5791:	vpand  ymm4,ymm10,ymm5
   d5795:	vpunpckldq ymm5,ymm6,ymm4
   d5799:	vpunpckhdq ymm9,ymm6,ymm4
   d579d:	vpand  ymm4,ymm10,YMMWORD PTR [rsp+0x80]
   d57a6:	vpand  ymm6,ymm10,ymm7
   d57aa:	vpunpckldq ymm7,ymm4,ymm6
   d57ae:	vpunpckhdq ymm4,ymm4,ymm6
   d57b2:	vpunpcklqdq ymm6,ymm8,ymm3
   d57b6:	vpunpckhqdq ymm3,ymm8,ymm3
   d57ba:	vpunpcklqdq ymm8,ymm0,ymm1
   d57be:	vpunpckhqdq ymm0,ymm0,ymm1
   d57c2:	vpunpcklqdq ymm1,ymm5,ymm7
   d57c6:	vpunpckhqdq ymm5,ymm5,ymm7
   d57ca:	vpunpcklqdq ymm7,ymm9,ymm4
   d57ce:	vpunpckhqdq ymm9,ymm9,ymm4
   d57d2:	vinserti128 ymm4,ymm6,xmm1,0x1
   d57d8:	vmovdqa YMMWORD PTR [rsp+0x240],ymm4
   d57e1:	vinserti128 ymm4,ymm3,xmm5,0x1
   d57e7:	vmovdqa YMMWORD PTR [rsp+0x260],ymm4
   d57f0:	vinserti128 ymm4,ymm8,xmm7,0x1
   d57f6:	vmovdqa YMMWORD PTR [rsp+0x280],ymm4
   d57ff:	vinserti128 ymm4,ymm0,xmm9,0x1
   d5805:	vmovdqa YMMWORD PTR [rsp+0x2a0],ymm4
   d580e:	vperm2i128 ymm1,ymm6,ymm1,0x31
   d5814:	vmovdqa YMMWORD PTR [rsp+0x2c0],ymm1
   d581d:	vperm2i128 ymm1,ymm3,ymm5,0x31
   d5823:	vmovdqa YMMWORD PTR [rsp+0x2e0],ymm1
   d582c:	vperm2i128 ymm1,ymm8,ymm7,0x31
   d5832:	vmovdqa YMMWORD PTR [rsp+0x300],ymm1
   d583b:	vperm2i128 ymm0,ymm0,ymm9,0x31
   d5841:	vmovdqa YMMWORD PTR [rsp+0x320],ymm0
   d584a:	vmovdqu ymm0,YMMWORD PTR [rsi+rcx*1+0x20]
   d5850:	vmovdqa YMMWORD PTR [rsp+0x80],ymm0
   d5859:	vmovdqu ymm1,YMMWORD PTR [rsi+rcx*1+0x60]
   d585f:	vmovdqu ymm11,YMMWORD PTR [rsi+rcx*1+0xa0]
   d5868:	vmovdqa YMMWORD PTR [rsp+0x60],ymm11
   d586e:	vmovdqu ymm3,YMMWORD PTR [rsi+rcx*1+0xe0]
   d5877:	vmovdqu ymm4,YMMWORD PTR [rdx+rcx*1+0x20]
   d587d:	vmovdqu ymm5,YMMWORD PTR [rdx+rcx*1+0x60]
   d5883:	vmovdqu ymm6,YMMWORD PTR [rdx+rcx*1+0xa0]
   d588c:	vmovdqu ymm7,YMMWORD PTR [rdx+rcx*1+0xe0]
   d5895:	vpand  ymm8,ymm0,ymm2
   d5899:	vpand  ymm9,ymm1,ymm2
   d589d:	vpunpckldq ymm10,ymm8,ymm9
   d58a2:	vpunpckhdq ymm8,ymm8,ymm9
   d58a7:	vpand  ymm9,ymm11,ymm2
   d58ab:	vpand  ymm11,ymm3,ymm2
   d58af:	vpunpckldq ymm12,ymm9,ymm11
   d58b4:	vpunpckhdq ymm9,ymm9,ymm11
   d58b9:	vpand  ymm11,ymm4,ymm2
   d58bd:	vpand  ymm13,ymm5,ymm2
   d58c1:	vpunpckldq ymm14,ymm11,ymm13
   d58c6:	vpunpckhdq ymm11,ymm11,ymm13
   d58cb:	vpand  ymm13,ymm6,ymm2
   d58cf:	vpand  ymm15,ymm7,ymm2
   d58d3:	vpunpckldq ymm0,ymm13,ymm15
   d58d8:	vpunpckhdq ymm13,ymm13,ymm15
   d58dd:	vpunpcklqdq ymm15,ymm10,ymm12
   d58e2:	vpunpckhqdq ymm10,ymm10,ymm12
   d58e7:	vpunpcklqdq ymm12,ymm8,ymm9
   d58ec:	vpunpckhqdq ymm8,ymm8,ymm9
   d58f1:	vpunpcklqdq ymm9,ymm14,ymm0
   d58f5:	vpunpckhqdq ymm0,ymm14,ymm0
   d58f9:	vpunpcklqdq ymm14,ymm11,ymm13
   d58fe:	vpunpckhqdq ymm11,ymm11,ymm13
   d5903:	vinserti128 ymm13,ymm15,xmm9,0x1
   d5909:	vmovdqa YMMWORD PTR [rsp+0x340],ymm13
   d5912:	vinserti128 ymm13,ymm10,xmm0,0x1
   d5918:	vmovdqa YMMWORD PTR [rsp+0x360],ymm13
   d5921:	vinserti128 ymm13,ymm12,xmm14,0x1
   d5927:	vmovdqa YMMWORD PTR [rsp+0x380],ymm13
   d5930:	vinserti128 ymm13,ymm8,xmm11,0x1
   d5936:	vmovdqa YMMWORD PTR [rsp+0x3a0],ymm13
   d593f:	vperm2i128 ymm9,ymm15,ymm9,0x31
   d5945:	vmovdqa YMMWORD PTR [rsp+0x3c0],ymm9
   d594e:	vperm2i128 ymm0,ymm10,ymm0,0x31
   d5954:	vmovdqa YMMWORD PTR [rsp+0x3e0],ymm0
   d595d:	vperm2i128 ymm0,ymm12,ymm14,0x31
   d5963:	vmovdqa YMMWORD PTR [rsp+0x400],ymm0
   d596c:	vperm2i128 ymm0,ymm8,ymm11,0x31
   d5972:	vmovdqa ymm2,YMMWORD PTR [rsp+0x80]
   d597b:	vpsrlw ymm8,ymm2,0x4
   d5980:	vpsrlw ymm1,ymm1,0x4
   d5985:	vmovdqa ymm2,YMMWORD PTR [rsp+0x60]
   d598b:	vpsrlw ymm2,ymm2,0x4
   d5990:	vpsrlw ymm3,ymm3,0x4
   d5995:	vpsrlw ymm4,ymm4,0x4
   d599a:	vpsrlw ymm5,ymm5,0x4
   d599f:	vpsrlw ymm6,ymm6,0x4
   d59a4:	vpsrlw ymm7,ymm7,0x4
   d59a9:	vmovdqa YMMWORD PTR [rsp+0x420],ymm0
   d59b2:	vpbroadcastb ymm9,BYTE PTR [rip+0x6e8f9]        # 1442b4 <_fini+0x1b94>
   d59bb:	vpand  ymm0,ymm8,ymm9
   d59c0:	vpand  ymm1,ymm9,ymm1
   d59c4:	vpunpckldq ymm8,ymm0,ymm1
   d59c8:	vpunpckhdq ymm0,ymm0,ymm1
   d59cc:	vpand  ymm1,ymm9,ymm2
   d59d0:	vpand  ymm2,ymm9,ymm3
   d59d4:	vpunpckldq ymm3,ymm1,ymm2
   d59d8:	vpunpckhdq ymm1,ymm1,ymm2
   d59dc:	vpand  ymm2,ymm9,ymm4
   d59e0:	vpand  ymm4,ymm9,ymm5
   d59e4:	vpunpckldq ymm5,ymm2,ymm4
   d59e8:	vpunpckhdq ymm2,ymm2,ymm4
   d59ec:	vpand  ymm4,ymm9,ymm6
   d59f0:	vpand  ymm6,ymm9,ymm7
   d59f4:	vpunpckldq ymm7,ymm4,ymm6
   d59f8:	vpunpckhdq ymm4,ymm4,ymm6
   d59fc:	vpunpcklqdq ymm6,ymm8,ymm3
   d5a00:	vpunpckhqdq ymm3,ymm8,ymm3
   d5a04:	vpunpcklqdq ymm8,ymm0,ymm1
   d5a08:	vpunpckhqdq ymm0,ymm0,ymm1
   d5a0c:	vpunpcklqdq ymm1,ymm5,ymm7
   d5a10:	vpunpckhqdq ymm5,ymm5,ymm7
   d5a14:	vpunpcklqdq ymm7,ymm2,ymm4
   d5a18:	vpunpckhqdq ymm2,ymm2,ymm4
   d5a1c:	vinserti128 ymm4,ymm6,xmm1,0x1
   d5a22:	vmovdqa YMMWORD PTR [rsp+0x440],ymm4
   d5a2b:	vinserti128 ymm4,ymm3,xmm5,0x1
   d5a31:	vmovdqa YMMWORD PTR [rsp+0x460],ymm4
   d5a3a:	vinserti128 ymm4,ymm8,xmm7,0x1
   d5a40:	vmovdqa YMMWORD PTR [rsp+0x480],ymm4
   d5a49:	vinserti128 ymm4,ymm0,xmm2,0x1
   d5a4f:	vmovdqa YMMWORD PTR [rsp+0x4a0],ymm4
   d5a58:	vperm2i128 ymm1,ymm6,ymm1,0x31
   d5a5e:	vmovdqa YMMWORD PTR [rsp+0x4c0],ymm1
   d5a67:	vperm2i128 ymm1,ymm3,ymm5,0x31
   d5a6d:	vmovdqa YMMWORD PTR [rsp+0x4e0],ymm1
   d5a76:	vperm2i128 ymm1,ymm8,ymm7,0x31
   d5a7c:	vmovdqa YMMWORD PTR [rsp+0x500],ymm1
   d5a85:	vperm2i128 ymm0,ymm0,ymm2,0x31
   d5a8b:	vmovdqa YMMWORD PTR [rsp+0x520],ymm0
   d5a94:	lea    rcx,[rax+0x1]
   d5a98:	test   r10,r10
   d5a9b:	je     d6c90 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1ba0>
   d5aa1:	mov    QWORD PTR [rsp+0x108],rcx
   d5aa9:	mov    rcx,rax
   d5aac:	shl    rcx,0x4
   d5ab0:	mov    rdx,QWORD PTR [rsp+0xf0]
   d5ab8:	vmovups xmm0,XMMWORD PTR [rdx+rcx*1]
   d5abd:	mov    rdx,QWORD PTR [rsp+0xe8]
   d5ac5:	vinsertf128 ymm0,ymm0,XMMWORD PTR [rdx+rcx*1],0x1
   d5acc:	mov    r9,rax
   d5acf:	shl    r9,0x7
   d5ad3:	add    r9,QWORD PTR [rsp+0x50]
   d5ad8:	mov    rcx,QWORD PTR [rsp+0xe0]
   d5ae0:	lea    rcx,[rcx+rax*4]
   d5ae4:	mov    QWORD PTR [rsp+0x120],rcx
   d5aec:	mov    rcx,QWORD PTR [rsp+0xd8]
   d5af4:	lea    rax,[rcx+rax*4]
   d5af8:	mov    QWORD PTR [rsp+0x28],rax
   d5afd:	vmovaps ymm1,YMMWORD PTR [rsp+0x140]
   d5b06:	vmovaps YMMWORD PTR [rsp+0x940],ymm1
   d5b0f:	vmovaps ymm1,YMMWORD PTR [rsp+0x160]
   d5b18:	vmovaps YMMWORD PTR [rsp+0x920],ymm1
   d5b21:	vmovaps ymm1,YMMWORD PTR [rsp+0x180]
   d5b2a:	vmovaps YMMWORD PTR [rsp+0x900],ymm1
   d5b33:	vmovaps ymm1,YMMWORD PTR [rsp+0x1a0]
   d5b3c:	vmovaps YMMWORD PTR [rsp+0x8e0],ymm1
   d5b45:	vmovaps ymm1,YMMWORD PTR [rsp+0x1c0]
   d5b4e:	vmovaps YMMWORD PTR [rsp+0x8c0],ymm1
   d5b57:	vmovaps ymm1,YMMWORD PTR [rsp+0x1e0]
   d5b60:	vmovaps YMMWORD PTR [rsp+0x8a0],ymm1
   d5b69:	vmovaps ymm1,YMMWORD PTR [rsp+0x200]
   d5b72:	vmovaps YMMWORD PTR [rsp+0x880],ymm1
   d5b7b:	vmovaps ymm1,YMMWORD PTR [rsp+0x220]
   d5b84:	vmovaps YMMWORD PTR [rsp+0x860],ymm1
   d5b8d:	vmovaps ymm1,YMMWORD PTR [rsp+0x240]
   d5b96:	vmovaps YMMWORD PTR [rsp+0x840],ymm1
   d5b9f:	vmovaps ymm1,YMMWORD PTR [rsp+0x260]
   d5ba8:	vmovaps YMMWORD PTR [rsp+0x820],ymm1
   d5bb1:	vmovaps ymm1,YMMWORD PTR [rsp+0x280]
   d5bba:	vmovaps YMMWORD PTR [rsp+0x800],ymm1
   d5bc3:	vmovaps ymm1,YMMWORD PTR [rsp+0x2a0]
   d5bcc:	vmovaps YMMWORD PTR [rsp+0x7e0],ymm1
   d5bd5:	vmovaps ymm1,YMMWORD PTR [rsp+0x2c0]
   d5bde:	vmovaps YMMWORD PTR [rsp+0x7c0],ymm1
   d5be7:	vmovaps ymm1,YMMWORD PTR [rsp+0x2e0]
   d5bf0:	vmovaps YMMWORD PTR [rsp+0x7a0],ymm1
   d5bf9:	vmovaps ymm1,YMMWORD PTR [rsp+0x300]
   d5c02:	vmovaps YMMWORD PTR [rsp+0x780],ymm1
   d5c0b:	xor    r13d,r13d
   d5c0e:	vmovaps ymm1,YMMWORD PTR [rsp+0x320]
   d5c17:	vmovaps YMMWORD PTR [rsp+0x760],ymm1
   d5c20:	vmovaps ymm1,YMMWORD PTR [rsp+0x340]
   d5c29:	vmovaps YMMWORD PTR [rsp+0x740],ymm1
   d5c32:	vmovaps ymm1,YMMWORD PTR [rsp+0x360]
   d5c3b:	vmovaps YMMWORD PTR [rsp+0x720],ymm1
   d5c44:	vmovaps ymm1,YMMWORD PTR [rsp+0x380]
   d5c4d:	vmovaps YMMWORD PTR [rsp+0x700],ymm1
   d5c56:	vmovaps ymm1,YMMWORD PTR [rsp+0x3a0]
   d5c5f:	vmovaps YMMWORD PTR [rsp+0x6e0],ymm1
   d5c68:	vmovaps ymm1,YMMWORD PTR [rsp+0x3c0]
   d5c71:	vmovaps YMMWORD PTR [rsp+0x6c0],ymm1
   d5c7a:	vmovaps ymm1,YMMWORD PTR [rsp+0x3e0]
   d5c83:	vmovaps YMMWORD PTR [rsp+0x6a0],ymm1
   d5c8c:	vmovaps ymm1,YMMWORD PTR [rsp+0x400]
   d5c95:	vmovaps YMMWORD PTR [rsp+0x680],ymm1
   d5c9e:	vmovaps ymm1,YMMWORD PTR [rsp+0x420]
   d5ca7:	vmovaps YMMWORD PTR [rsp+0x660],ymm1
   d5cb0:	vmovaps ymm1,YMMWORD PTR [rsp+0x440]
   d5cb9:	vmovaps YMMWORD PTR [rsp+0x640],ymm1
   d5cc2:	vmovaps ymm1,YMMWORD PTR [rsp+0x460]
   d5ccb:	vmovaps YMMWORD PTR [rsp+0x620],ymm1
   d5cd4:	vmovaps ymm1,YMMWORD PTR [rsp+0x480]
   d5cdd:	vmovaps YMMWORD PTR [rsp+0x600],ymm1
   d5ce6:	vmovaps ymm1,YMMWORD PTR [rsp+0x4a0]
   d5cef:	vmovaps YMMWORD PTR [rsp+0x5e0],ymm1
   d5cf8:	vmovaps ymm1,YMMWORD PTR [rsp+0x4c0]
   d5d01:	vmovaps YMMWORD PTR [rsp+0x5c0],ymm1
   d5d0a:	vmovaps ymm1,YMMWORD PTR [rsp+0x4e0]
   d5d13:	vmovaps YMMWORD PTR [rsp+0x5a0],ymm1
   d5d1c:	vmovaps ymm1,YMMWORD PTR [rsp+0x500]
   d5d25:	vmovaps YMMWORD PTR [rsp+0x580],ymm1
   d5d2e:	vmovdqa ymm1,YMMWORD PTR [rsp+0x520]
   d5d37:	vmovdqa YMMWORD PTR [rsp+0x560],ymm1
   d5d40:	vpbroadcastw ymm14,WORD PTR [rip+0x6e56d]        # 1442b6 <_fini+0x1b96>
   d5d49:	jmp    d5d61 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc71>
   d5d4b:	nop    DWORD PTR [rax+rax*1+0x0]
   d5d50:	add    r13,QWORD PTR [rsp+0x80]
   d5d58:	cmp    r13,r10
   d5d5b:	jae    d5590 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x4a0>
   d5d61:	mov    rax,r10
   d5d64:	sub    rax,r13
   d5d67:	cmp    rax,0x4
   d5d6b:	mov    ecx,0x4
   d5d70:	mov    QWORD PTR [rsp+0x60],rax
   d5d75:	cmovb  rcx,rax
   d5d79:	cmp    rcx,0x2
   d5d7d:	mov    QWORD PTR [rsp+0x80],rcx
   d5d85:	je     d6470 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1380>
   d5d8b:	cmp    rcx,0x3
   d5d8f:	je     d6180 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1090>
   d5d95:	cmp    rcx,0x4
   d5d99:	jne    d6690 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x15a0>
   d5d9f:	mov    r10,r11
   d5da2:	cmp    r13,r11
   d5da5:	jae    d6f02 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1e12>
   d5dab:	mov    r11,r15
   d5dae:	mov    rax,QWORD PTR [rsp+0x18]
   d5db3:	movzx  r15d,WORD PTR [rax+r13*4]
   d5db8:	mov    eax,r15d
   d5dbb:	shl    eax,0xc
   d5dbe:	lea    rsi,[r15+0x1]
   d5dc2:	mov    edx,esi
   d5dc4:	shl    edx,0xc
   d5dc7:	cmp    rdx,r12
   d5dca:	ja     d6e24 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d34>
   d5dd0:	shl    esi,0x5
   d5dd3:	cmp    rsi,r11
   d5dd6:	ja     d6db5 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc5>
   d5ddc:	cmp    rsi,rbx
   d5ddf:	ja     d6dce <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cde>
   d5de5:	lea    rcx,[r13+0x1]
   d5de9:	cmp    rcx,r10
   d5dec:	jae    d6f05 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1e15>
   d5df2:	mov    rcx,QWORD PTR [rsp+0x18]
   d5df7:	movzx  r15d,WORD PTR [rcx+r13*4+0x4]
   d5dfd:	mov    r8d,r15d
   d5e00:	shl    r8d,0xc
   d5e04:	lea    rsi,[r15+0x1]
   d5e08:	mov    edx,esi
   d5e0a:	shl    edx,0xc
   d5e0d:	cmp    rdx,r12
   d5e10:	ja     d6e21 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d31>
   d5e16:	shl    esi,0x5
   d5e19:	cmp    rsi,r11
   d5e1c:	ja     d6db5 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc5>
   d5e22:	cmp    rsi,rbx
   d5e25:	ja     d6dce <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cde>
   d5e2b:	lea    rcx,[r13+0x2]
   d5e2f:	cmp    rcx,r10
   d5e32:	jae    d6f05 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1e15>
   d5e38:	mov    rcx,QWORD PTR [rsp+0x18]
   d5e3d:	movzx  r15d,WORD PTR [rcx+r13*4+0x8]
   d5e43:	mov    edi,r15d
   d5e46:	shl    edi,0xc
   d5e49:	lea    rsi,[r15+0x1]
   d5e4d:	mov    edx,esi
   d5e4f:	shl    edx,0xc
   d5e52:	cmp    rdx,r12
   d5e55:	ja     d6dec <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cfc>
   d5e5b:	shl    esi,0x5
   d5e5e:	cmp    rsi,r11
   d5e61:	ja     d6db5 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc5>
   d5e67:	cmp    rsi,rbx
   d5e6a:	ja     d6dce <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cde>
   d5e70:	lea    rcx,[r13+0x3]
   d5e74:	cmp    rcx,r10
   d5e77:	jae    d6f05 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1e15>
   d5e7d:	mov    rcx,QWORD PTR [rsp+0x18]
   d5e82:	movzx  r15d,WORD PTR [rcx+r13*4+0xc]
   d5e88:	mov    ecx,r15d
   d5e8b:	shl    ecx,0xc
   d5e8e:	lea    rsi,[r15+0x1]
   d5e92:	mov    edx,esi
   d5e94:	shl    edx,0xc
   d5e97:	cmp    rdx,r12
   d5e9a:	ja     d6de7 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cf7>
   d5ea0:	shl    esi,0x5
   d5ea3:	cmp    rsi,r11
   d5ea6:	ja     d6db5 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc5>
   d5eac:	cmp    rsi,rbx
   d5eaf:	ja     d6dce <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cde>
   d5eb5:	mov    rdx,QWORD PTR [rsp+0x20]
   d5eba:	add    rax,rdx
   d5ebd:	add    r8,rdx
   d5ec0:	add    rdi,rdx
   d5ec3:	add    rcx,QWORD PTR [rsp+0x48]
   d5ec8:	vpxor  xmm3,xmm3,xmm3
   d5ecc:	xor    edx,edx
   d5ece:	vpxor  xmm5,xmm5,xmm5
   d5ed2:	vpxor  xmm4,xmm4,xmm4
   d5ed6:	vpxor  xmm6,xmm6,xmm6
   d5eda:	mov    r15,r11
   d5edd:	mov    r11,r10
   d5ee0:	mov    r10,QWORD PTR [rsp+0x40]
   d5ee5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   d5ef0:	vpbroadcastd ymm8,DWORD PTR [rax+rdx*1-0x1c]
   d5ef7:	vmovdqa ymm9,YMMWORD PTR [rsp+rdx*8+0x140]
   d5f00:	vmovdqa ymm10,YMMWORD PTR [rsp+rdx*8+0x160]
   d5f09:	vmovdqa ymm11,YMMWORD PTR [rsp+rdx*8+0x180]
   d5f12:	vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x1a0]
   d5f1b:	vpmaddubsw ymm8,ymm9,ymm8
   d5f20:	vpbroadcastd ymm12,DWORD PTR [r8+rdx*1-0x1c]
   d5f27:	vpmaddubsw ymm12,ymm9,ymm12
   d5f2c:	vpbroadcastd ymm13,DWORD PTR [rdi+rdx*1-0x1c]
   d5f33:	vpbroadcastd ymm14,DWORD PTR [rcx+rdx*1]
   d5f39:	vpmaddubsw ymm13,ymm9,ymm13
   d5f3e:	vpmaddubsw ymm9,ymm9,ymm14
   d5f43:	vpbroadcastd ymm14,DWORD PTR [rax+rdx*1-0x18]
   d5f4a:	vpmaddubsw ymm14,ymm10,ymm14
   d5f4f:	vpaddw ymm8,ymm14,ymm8
   d5f54:	vpbroadcastd ymm14,DWORD PTR [r8+rdx*1-0x18]
   d5f5b:	vpmaddubsw ymm14,ymm10,ymm14
   d5f60:	vpaddw ymm12,ymm14,ymm12
   d5f65:	vpbroadcastd ymm14,DWORD PTR [rdi+rdx*1-0x18]
   d5f6c:	vpmaddubsw ymm14,ymm10,ymm14
   d5f71:	vpaddw ymm13,ymm14,ymm13
   d5f76:	vpbroadcastd ymm14,DWORD PTR [rcx+rdx*1+0x4]
   d5f7d:	vpmaddubsw ymm10,ymm10,ymm14
   d5f82:	vpaddw ymm14,ymm10,ymm9
   d5f87:	vpbroadcastd ymm9,DWORD PTR [rax+rdx*1-0x14]
   d5f8e:	vpmaddubsw ymm9,ymm11,ymm9
   d5f93:	vpbroadcastd ymm10,DWORD PTR [r8+rdx*1-0x14]
   d5f9a:	vpmaddubsw ymm15,ymm11,ymm10
   d5f9f:	vpbroadcastd ymm10,DWORD PTR [rdi+rdx*1-0x14]
   d5fa6:	vpbroadcastd ymm1,DWORD PTR [rcx+rdx*1+0x8]
   d5fad:	vpmaddubsw ymm2,ymm11,ymm10
   d5fb2:	vpmaddubsw ymm1,ymm11,ymm1
   d5fb7:	vpbroadcastd ymm10,DWORD PTR [rax+rdx*1-0x10]
   d5fbe:	vpmaddubsw ymm10,ymm7,ymm10
   d5fc3:	vpaddw ymm9,ymm10,ymm9
   d5fc8:	vpbroadcastd ymm11,DWORD PTR [r8+rdx*1-0x10]
   d5fcf:	vpaddw ymm10,ymm9,ymm8
   d5fd4:	vpmaddubsw ymm8,ymm7,ymm11
   d5fd9:	vpaddw ymm8,ymm8,ymm15
   d5fde:	vpaddw ymm8,ymm8,ymm12
   d5fe3:	vpbroadcastd ymm9,DWORD PTR [rdi+rdx*1-0x10]
   d5fea:	vpmaddubsw ymm9,ymm7,ymm9
   d5fef:	vpaddw ymm2,ymm9,ymm2
   d5ff3:	vpaddw ymm9,ymm13,ymm2
   d5ff7:	vpbroadcastd ymm2,DWORD PTR [rcx+rdx*1+0xc]
   d5ffe:	vpmaddubsw ymm2,ymm7,ymm2
   d6003:	vpaddw ymm1,ymm2,ymm1
   d6007:	vpaddw ymm7,ymm14,ymm1
   d600b:	vpbroadcastd ymm1,DWORD PTR [rax+rdx*1-0xc]
   d6012:	vmovdqa ymm2,YMMWORD PTR [rsp+rdx*8+0x1c0]
   d601b:	vpmaddubsw ymm1,ymm2,ymm1
   d6020:	vpbroadcastd ymm11,DWORD PTR [r8+rdx*1-0xc]
   d6027:	vpmaddubsw ymm11,ymm2,ymm11
   d602c:	vpbroadcastd ymm12,DWORD PTR [rdi+rdx*1-0xc]
   d6033:	vpmaddubsw ymm12,ymm2,ymm12
   d6038:	vpbroadcastd ymm13,DWORD PTR [rcx+rdx*1+0x10]
   d603f:	vpmaddubsw ymm2,ymm2,ymm13
   d6044:	vmovdqa ymm13,YMMWORD PTR [rsp+rdx*8+0x1e0]
   d604d:	vpbroadcastd ymm14,DWORD PTR [rax+rdx*1-0x8]
   d6054:	vpmaddubsw ymm14,ymm13,ymm14
   d6059:	vpaddw ymm1,ymm14,ymm1
   d605d:	vpbroadcastd ymm14,DWORD PTR [r8+rdx*1-0x8]
   d6064:	vpmaddubsw ymm14,ymm13,ymm14
   d6069:	vpaddw ymm11,ymm14,ymm11
   d606e:	vpbroadcastd ymm14,DWORD PTR [rdi+rdx*1-0x8]
   d6075:	vpmaddubsw ymm14,ymm13,ymm14
   d607a:	vpaddw ymm12,ymm14,ymm12
   d607f:	vpbroadcastd ymm14,DWORD PTR [rcx+rdx*1+0x14]
   d6086:	vpmaddubsw ymm13,ymm13,ymm14
   d608b:	vpaddw ymm2,ymm13,ymm2
   d608f:	vmovdqa ymm13,YMMWORD PTR [rsp+rdx*8+0x200]
   d6098:	vpbroadcastd ymm14,DWORD PTR [rax+rdx*1-0x4]
   d609f:	vpmaddubsw ymm14,ymm13,ymm14
   d60a4:	vpaddw ymm1,ymm14,ymm1
   d60a8:	vpbroadcastw ymm14,WORD PTR [rip+0x6e205]        # 1442b6 <_fini+0x1b96>
   d60b1:	vpaddw ymm1,ymm10,ymm1
   d60b5:	vpbroadcastd ymm10,DWORD PTR [r8+rdx*1-0x4]
   d60bc:	vpmaddubsw ymm10,ymm13,ymm10
   d60c1:	vpaddw ymm10,ymm10,ymm11
   d60c6:	vpbroadcastd ymm11,DWORD PTR [rdi+rdx*1-0x4]
   d60cd:	vpaddw ymm8,ymm10,ymm8
   d60d2:	vpmaddubsw ymm10,ymm13,ymm11
   d60d7:	vpaddw ymm10,ymm10,ymm12
   d60dc:	vpaddw ymm9,ymm10,ymm9
   d60e1:	vpbroadcastd ymm10,DWORD PTR [rcx+rdx*1+0x18]
   d60e8:	vpmaddubsw ymm10,ymm13,ymm10
   d60ed:	vpaddw ymm2,ymm10,ymm2
   d60f1:	vpaddw ymm2,ymm2,ymm7
   d60f5:	vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x220]
   d60fe:	vpbroadcastd ymm10,DWORD PTR [rax+rdx*1]
   d6104:	vpmaddubsw ymm10,ymm7,ymm10
   d6109:	vpaddw ymm1,ymm10,ymm1
   d610d:	vpbroadcastd ymm10,DWORD PTR [r8+rdx*1]
   d6113:	vpmaddubsw ymm10,ymm7,ymm10
   d6118:	vpaddw ymm8,ymm10,ymm8
   d611d:	vpbroadcastd ymm10,DWORD PTR [rdi+rdx*1]
   d6123:	vpmaddubsw ymm10,ymm7,ymm10
   d6128:	vpaddw ymm9,ymm10,ymm9
   d612d:	vpbroadcastd ymm10,DWORD PTR [rcx+rdx*1+0x1c]
   d6134:	vpmaddubsw ymm7,ymm7,ymm10
   d6139:	vpaddw ymm2,ymm7,ymm2
   d613d:	vpmaddwd ymm1,ymm14,ymm1
   d6141:	vpaddd ymm6,ymm1,ymm6
   d6145:	vpmaddwd ymm1,ymm8,ymm14
   d614a:	vpaddd ymm4,ymm1,ymm4
   d614e:	vpmaddwd ymm1,ymm9,ymm14
   d6153:	vpaddd ymm5,ymm1,ymm5
   d6157:	vpmaddwd ymm1,ymm14,ymm2
   d615b:	vpaddd ymm3,ymm1,ymm3
   d615f:	add    rdx,0x20
   d6163:	cmp    rdx,0x80
   d616a:	jne    d5ef0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xe00>
   d6170:	jmp    d6454 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1364>
   d6175:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   d6180:	cmp    r13,r11
   d6183:	jae    d6eb8 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dc8>
   d6189:	mov    rax,QWORD PTR [rsp+0x18]
   d618e:	movzx  r8d,WORD PTR [rax+r13*4]
   d6193:	mov    eax,r8d
   d6196:	shl    eax,0xc
   d6199:	lea    rsi,[r8+0x1]
   d619d:	mov    edx,esi
   d619f:	shl    edx,0xc
   d61a2:	cmp    rdx,r12
   d61a5:	ja     d6e24 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d34>
   d61ab:	shl    esi,0x5
   d61ae:	cmp    rsi,r15
   d61b1:	ja     d6d7e <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c8e>
   d61b7:	cmp    rsi,rbx
   d61ba:	ja     d6d97 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1ca7>
   d61c0:	lea    rcx,[r13+0x1]
   d61c4:	cmp    rcx,r11
   d61c7:	jae    d6ebb <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dcb>
   d61cd:	mov    rcx,QWORD PTR [rsp+0x18]
   d61d2:	movzx  r8d,WORD PTR [rcx+r13*4+0x4]
   d61d8:	mov    edi,r8d
   d61db:	shl    edi,0xc
   d61de:	lea    rsi,[r8+0x1]
   d61e2:	mov    edx,esi
   d61e4:	shl    edx,0xc
   d61e7:	cmp    rdx,r12
   d61ea:	ja     d6dec <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cfc>
   d61f0:	shl    esi,0x5
   d61f3:	cmp    rsi,r15
   d61f6:	ja     d6d7e <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c8e>
   d61fc:	cmp    rsi,rbx
   d61ff:	ja     d6d97 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1ca7>
   d6205:	lea    rcx,[r13+0x2]
   d6209:	cmp    rcx,r11
   d620c:	jae    d6ebb <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dcb>
   d6212:	mov    rcx,QWORD PTR [rsp+0x18]
   d6217:	movzx  r8d,WORD PTR [rcx+r13*4+0x8]
   d621d:	mov    ecx,r8d
   d6220:	shl    ecx,0xc
   d6223:	lea    rsi,[r8+0x1]
   d6227:	mov    edx,esi
   d6229:	shl    edx,0xc
   d622c:	cmp    rdx,r12
   d622f:	ja     d6de7 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cf7>
   d6235:	shl    esi,0x5
   d6238:	cmp    rsi,r15
   d623b:	ja     d6d7e <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c8e>
   d6241:	cmp    rsi,rbx
   d6244:	ja     d6d97 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1ca7>
   d624a:	mov    rdx,QWORD PTR [rsp+0x20]
   d624f:	add    rax,rdx
   d6252:	add    rdi,rdx
   d6255:	add    rcx,QWORD PTR [rsp+0x48]
   d625a:	vpxor  xmm3,xmm3,xmm3
   d625e:	xor    edx,edx
   d6260:	vpxor  xmm5,xmm5,xmm5
   d6264:	vpxor  xmm4,xmm4,xmm4
   d6268:	vpxor  xmm6,xmm6,xmm6
   d626c:	nop    DWORD PTR [rax+0x0]
   d6270:	vpbroadcastd ymm1,DWORD PTR [rax+rdx*1-0x1c]
   d6277:	vmovdqa ymm2,YMMWORD PTR [rsp+rdx*8+0x140]
   d6280:	vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x160]
   d6289:	vmovdqa ymm8,YMMWORD PTR [rsp+rdx*8+0x180]
   d6292:	vmovdqa ymm10,YMMWORD PTR [rsp+rdx*8+0x1a0]
   d629b:	vpmaddubsw ymm1,ymm2,ymm1
   d62a0:	vpbroadcastd ymm9,DWORD PTR [rdi+rdx*1-0x1c]
   d62a7:	vpmaddubsw ymm9,ymm2,ymm9
   d62ac:	vpbroadcastd ymm11,DWORD PTR [rcx+rdx*1]
   d62b2:	vpmaddubsw ymm2,ymm2,ymm11
   d62b7:	vpbroadcastd ymm11,DWORD PTR [rax+rdx*1-0x18]
   d62be:	vpmaddubsw ymm11,ymm7,ymm11
   d62c3:	vpaddw ymm1,ymm11,ymm1
   d62c7:	vpbroadcastd ymm11,DWORD PTR [rdi+rdx*1-0x18]
   d62ce:	vpmaddubsw ymm11,ymm7,ymm11
   d62d3:	vpaddw ymm9,ymm11,ymm9
   d62d8:	vpbroadcastd ymm11,DWORD PTR [rcx+rdx*1+0x4]
   d62df:	vpmaddubsw ymm7,ymm7,ymm11
   d62e4:	vpaddw ymm2,ymm7,ymm2
   d62e8:	vpbroadcastd ymm7,DWORD PTR [rax+rdx*1-0x14]
   d62ef:	vpbroadcastd ymm11,DWORD PTR [rdi+rdx*1-0x14]
   d62f6:	vpmaddubsw ymm7,ymm8,ymm7
   d62fb:	vpmaddubsw ymm11,ymm8,ymm11
   d6300:	vpbroadcastd ymm12,DWORD PTR [rcx+rdx*1+0x8]
   d6307:	vpmaddubsw ymm8,ymm8,ymm12
   d630c:	vpbroadcastd ymm12,DWORD PTR [rax+rdx*1-0x10]
   d6313:	vpmaddubsw ymm12,ymm10,ymm12
   d6318:	vpaddw ymm7,ymm12,ymm7
   d631c:	vpaddw ymm7,ymm7,ymm1
   d6320:	vpbroadcastd ymm1,DWORD PTR [rdi+rdx*1-0x10]
   d6327:	vpmaddubsw ymm1,ymm10,ymm1
   d632c:	vpaddw ymm1,ymm11,ymm1
   d6330:	vpbroadcastd ymm11,DWORD PTR [rcx+rdx*1+0xc]
   d6337:	vpaddw ymm9,ymm9,ymm1
   d633b:	vpmaddubsw ymm1,ymm10,ymm11
   d6340:	vpaddw ymm1,ymm8,ymm1
   d6344:	vpaddw ymm8,ymm1,ymm2
   d6348:	vmovdqa ymm1,YMMWORD PTR [rsp+rdx*8+0x1c0]
   d6351:	vpbroadcastd ymm2,DWORD PTR [rax+rdx*1-0xc]
   d6358:	vpmaddubsw ymm2,ymm1,ymm2
   d635d:	vpbroadcastd ymm10,DWORD PTR [rdi+rdx*1-0xc]
   d6364:	vpbroadcastd ymm11,DWORD PTR [rcx+rdx*1+0x10]
   d636b:	vpmaddubsw ymm10,ymm1,ymm10
   d6370:	vpmaddubsw ymm1,ymm1,ymm11
   d6375:	vpbroadcastd ymm11,DWORD PTR [rax+rdx*1-0x8]
   d637c:	vmovdqa ymm12,YMMWORD PTR [rsp+rdx*8+0x1e0]
   d6385:	vpmaddubsw ymm11,ymm12,ymm11
   d638a:	vpbroadcastd ymm13,DWORD PTR [rdi+rdx*1-0x8]
   d6391:	vpaddw ymm2,ymm11,ymm2
   d6395:	vpmaddubsw ymm11,ymm12,ymm13
   d639a:	vpbroadcastd ymm13,DWORD PTR [rcx+rdx*1+0x14]
   d63a1:	vpaddw ymm10,ymm11,ymm10
   d63a6:	vpmaddubsw ymm11,ymm12,ymm13
   d63ab:	vpaddw ymm1,ymm11,ymm1
   d63af:	vmovdqa ymm11,YMMWORD PTR [rsp+rdx*8+0x200]
   d63b8:	vpbroadcastd ymm12,DWORD PTR [rax+rdx*1-0x4]
   d63bf:	vpmaddubsw ymm12,ymm11,ymm12
   d63c4:	vpaddw ymm2,ymm12,ymm2
   d63c8:	vpaddw ymm2,ymm2,ymm7
   d63cc:	vpbroadcastd ymm7,DWORD PTR [rdi+rdx*1-0x4]
   d63d3:	vpmaddubsw ymm7,ymm11,ymm7
   d63d8:	vpaddw ymm7,ymm10,ymm7
   d63dc:	vpaddw ymm7,ymm9,ymm7
   d63e0:	vpbroadcastd ymm9,DWORD PTR [rcx+rdx*1+0x18]
   d63e7:	vpmaddubsw ymm9,ymm11,ymm9
   d63ec:	vpaddw ymm1,ymm9,ymm1
   d63f0:	vpaddw ymm1,ymm8,ymm1
   d63f4:	vmovdqa ymm8,YMMWORD PTR [rsp+rdx*8+0x220]
   d63fd:	vpbroadcastd ymm9,DWORD PTR [rax+rdx*1]
   d6403:	vpmaddubsw ymm9,ymm8,ymm9
   d6408:	vpaddw ymm2,ymm9,ymm2
   d640c:	vpbroadcastd ymm9,DWORD PTR [rdi+rdx*1]
   d6412:	vpmaddubsw ymm9,ymm8,ymm9
   d6417:	vpaddw ymm7,ymm9,ymm7
   d641b:	vpbroadcastd ymm9,DWORD PTR [rcx+rdx*1+0x1c]
   d6422:	vpmaddubsw ymm8,ymm8,ymm9
   d6427:	vpaddw ymm1,ymm8,ymm1
   d642b:	vpmaddwd ymm2,ymm14,ymm2
   d642f:	vpaddd ymm6,ymm2,ymm6
   d6433:	vpmaddwd ymm2,ymm14,ymm7
   d6437:	vpaddd ymm4,ymm2,ymm4
   d643b:	vpmaddwd ymm1,ymm14,ymm1
   d643f:	vpaddd ymm5,ymm1,ymm5
   d6443:	add    rdx,0x20
   d6447:	cmp    rdx,0x80
   d644e:	jne    d6270 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1180>
   d6454:	cmp    r10,r13
   d6457:	je     d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d645d:	jmp    d6a0d <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x191d>
   d6462:	data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
   d6470:	cmp    r13,r11
   d6473:	jae    d6eb8 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dc8>
   d6479:	mov    rax,QWORD PTR [rsp+0x18]
   d647e:	movzx  r8d,WORD PTR [rax+r13*4]
   d6483:	mov    eax,r8d
   d6486:	shl    eax,0xc
   d6489:	lea    rsi,[r8+0x1]
   d648d:	mov    edx,esi
   d648f:	shl    edx,0xc
   d6492:	cmp    rdx,r12
   d6495:	ja     d6e24 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d34>
   d649b:	shl    esi,0x5
   d649e:	cmp    rsi,r15
   d64a1:	ja     d6d7e <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c8e>
   d64a7:	cmp    rsi,rbx
   d64aa:	ja     d6d97 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1ca7>
   d64b0:	lea    rcx,[r13+0x1]
   d64b4:	cmp    rcx,r11
   d64b7:	jae    d6ebb <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dcb>
   d64bd:	mov    rcx,QWORD PTR [rsp+0x18]
   d64c2:	movzx  r8d,WORD PTR [rcx+r13*4+0x4]
   d64c8:	mov    ecx,r8d
   d64cb:	shl    ecx,0xc
   d64ce:	lea    rsi,[r8+0x1]
   d64d2:	mov    edx,esi
   d64d4:	shl    edx,0xc
   d64d7:	cmp    rdx,r12
   d64da:	ja     d6de7 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cf7>
   d64e0:	shl    esi,0x5
   d64e3:	cmp    rsi,r15
   d64e6:	ja     d6d7e <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c8e>
   d64ec:	cmp    rsi,rbx
   d64ef:	ja     d6d97 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1ca7>
   d64f5:	add    rax,QWORD PTR [rsp+0x20]
   d64fa:	add    rcx,QWORD PTR [rsp+0x58]
   d64ff:	vpxor  xmm3,xmm3,xmm3
   d6503:	xor    edx,edx
   d6505:	vpxor  xmm4,xmm4,xmm4
   d6509:	vpxor  xmm6,xmm6,xmm6
   d650d:	nop    DWORD PTR [rax]
   d6510:	vpbroadcastd ymm1,DWORD PTR [rax+rdx*1-0x1c]
   d6517:	vmovdqa ymm2,YMMWORD PTR [rsp+rdx*8+0x140]
   d6520:	vmovdqa ymm5,YMMWORD PTR [rsp+rdx*8+0x160]
   d6529:	vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x180]
   d6532:	vmovdqa ymm8,YMMWORD PTR [rsp+rdx*8+0x1a0]
   d653b:	vpbroadcastd ymm9,DWORD PTR [rcx+rdx*1-0x10]
   d6542:	vpmaddubsw ymm1,ymm2,ymm1
   d6547:	vpmaddubsw ymm2,ymm2,ymm9
   d654c:	vpbroadcastd ymm9,DWORD PTR [rax+rdx*1-0x18]
   d6553:	vpmaddubsw ymm9,ymm5,ymm9
   d6558:	vpaddw ymm1,ymm9,ymm1
   d655c:	vpbroadcastd ymm9,DWORD PTR [rcx+rdx*1-0xc]
   d6563:	vpmaddubsw ymm5,ymm5,ymm9
   d6568:	vpaddw ymm2,ymm5,ymm2
   d656c:	vpbroadcastd ymm5,DWORD PTR [rax+rdx*1-0x14]
   d6573:	vpmaddubsw ymm5,ymm7,ymm5
   d6578:	vpbroadcastd ymm9,DWORD PTR [rcx+rdx*1-0x8]
   d657f:	vpmaddubsw ymm7,ymm7,ymm9
   d6584:	vpbroadcastd ymm9,DWORD PTR [rax+rdx*1-0x10]
   d658b:	vpmaddubsw ymm9,ymm8,ymm9
   d6590:	vpaddw ymm5,ymm9,ymm5
   d6594:	vpaddw ymm1,ymm5,ymm1
   d6598:	vpbroadcastd ymm5,DWORD PTR [rcx+rdx*1-0x4]
   d659f:	vpmaddubsw ymm5,ymm8,ymm5
   d65a4:	vpaddw ymm5,ymm5,ymm7
   d65a8:	vpaddw ymm2,ymm5,ymm2
   d65ac:	vpbroadcastd ymm5,DWORD PTR [rax+rdx*1-0xc]
   d65b3:	vmovdqa ymm7,YMMWORD PTR [rsp+rdx*8+0x1c0]
   d65bc:	vpmaddubsw ymm5,ymm7,ymm5
   d65c1:	vpbroadcastd ymm8,DWORD PTR [rcx+rdx*1]
   d65c7:	vpmaddubsw ymm7,ymm7,ymm8
   d65cc:	vmovdqa ymm8,YMMWORD PTR [rsp+rdx*8+0x1e0]
   d65d5:	vpbroadcastd ymm9,DWORD PTR [rax+rdx*1-0x8]
   d65dc:	vpmaddubsw ymm9,ymm8,ymm9
   d65e1:	vpaddw ymm5,ymm9,ymm5
   d65e5:	vpbroadcastd ymm9,DWORD PTR [rcx+rdx*1+0x4]
   d65ec:	vpmaddubsw ymm8,ymm8,ymm9
   d65f1:	vpaddw ymm7,ymm8,ymm7
   d65f5:	vpbroadcastd ymm8,DWORD PTR [rax+rdx*1-0x4]
   d65fc:	vmovdqa ymm9,YMMWORD PTR [rsp+rdx*8+0x200]
   d6605:	vpmaddubsw ymm8,ymm9,ymm8
   d660a:	vpaddw ymm5,ymm8,ymm5
   d660e:	vpaddw ymm1,ymm5,ymm1
   d6612:	vpbroadcastd ymm5,DWORD PTR [rcx+rdx*1+0x8]
   d6619:	vpmaddubsw ymm5,ymm9,ymm5
   d661e:	vpaddw ymm5,ymm5,ymm7
   d6622:	vpaddw ymm2,ymm5,ymm2
   d6626:	vmovdqa ymm5,YMMWORD PTR [rsp+rdx*8+0x220]
   d662f:	vpbroadcastd ymm7,DWORD PTR [rax+rdx*1]
   d6635:	vpmaddubsw ymm7,ymm5,ymm7
   d663a:	vpaddw ymm1,ymm7,ymm1
   d663e:	vpbroadcastd ymm7,DWORD PTR [rcx+rdx*1+0xc]
   d6645:	vpmaddubsw ymm5,ymm5,ymm7
   d664a:	vpaddw ymm2,ymm5,ymm2
   d664e:	vpmaddwd ymm1,ymm14,ymm1
   d6652:	vpaddd ymm6,ymm1,ymm6
   d6656:	vpmaddwd ymm1,ymm14,ymm2
   d665a:	vpaddd ymm4,ymm1,ymm4
   d665e:	add    rdx,0x20
   d6662:	cmp    rdx,0x80
   d6669:	jne    d6510 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1420>
   d666f:	vpxor  xmm5,xmm5,xmm5
   d6673:	cmp    r10,r13
   d6676:	je     d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d667c:	jmp    d6a0d <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x191d>
   d6681:	data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
   d6690:	cmp    r13,r11
   d6693:	jae    d6ed3 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1de3>
   d6699:	mov    rax,QWORD PTR [rsp+0x18]
   d669e:	movzx  eax,WORD PTR [rax+r13*4]
   d66a3:	mov    esi,eax
   d66a5:	inc    esi
   d66a7:	mov    edx,esi
   d66a9:	shl    edx,0xc
   d66ac:	cmp    rdx,r12
   d66af:	ja     d6db0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc0>
   d66b5:	shl    esi,0x5
   d66b8:	cmp    rsi,r15
   d66bb:	ja     d6df1 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d01>
   d66c1:	cmp    rsi,rbx
   d66c4:	ja     d6e09 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d19>
   d66ca:	shl    eax,0xc
   d66cd:	vpbroadcastd ymm3,DWORD PTR [r9+rax*1]
   d66d3:	vmovdqa ymm1,YMMWORD PTR [rsp+0x940]
   d66dc:	vpmaddubsw ymm3,ymm1,ymm3
   d66e1:	vpbroadcastd ymm4,DWORD PTR [r9+rax*1+0x4]
   d66e8:	vmovdqa ymm1,YMMWORD PTR [rsp+0x920]
   d66f1:	vpmaddubsw ymm4,ymm1,ymm4
   d66f6:	vpaddw ymm3,ymm4,ymm3
   d66fa:	vpbroadcastd ymm4,DWORD PTR [r9+rax*1+0x8]
   d6701:	vmovdqa ymm1,YMMWORD PTR [rsp+0x900]
   d670a:	vpmaddubsw ymm4,ymm1,ymm4
   d670f:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0xc]
   d6716:	vmovdqa ymm1,YMMWORD PTR [rsp+0x8e0]
   d671f:	vpmaddubsw ymm5,ymm1,ymm5
   d6724:	vpaddw ymm4,ymm5,ymm4
   d6728:	vpaddw ymm3,ymm4,ymm3
   d672c:	vpbroadcastd ymm4,DWORD PTR [r9+rax*1+0x10]
   d6733:	vmovdqa ymm1,YMMWORD PTR [rsp+0x8c0]
   d673c:	vpmaddubsw ymm4,ymm1,ymm4
   d6741:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x14]
   d6748:	vmovdqa ymm1,YMMWORD PTR [rsp+0x8a0]
   d6751:	vpmaddubsw ymm5,ymm1,ymm5
   d6756:	vpaddw ymm4,ymm5,ymm4
   d675a:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x18]
   d6761:	vmovdqa ymm1,YMMWORD PTR [rsp+0x880]
   d676a:	vpmaddubsw ymm5,ymm1,ymm5
   d676f:	vpaddw ymm4,ymm5,ymm4
   d6773:	vpaddw ymm3,ymm4,ymm3
   d6777:	vpbroadcastd ymm4,DWORD PTR [r9+rax*1+0x1c]
   d677e:	vmovdqa ymm1,YMMWORD PTR [rsp+0x860]
   d6787:	vpmaddubsw ymm4,ymm1,ymm4
   d678c:	vpaddw ymm3,ymm4,ymm3
   d6790:	vpmaddwd ymm3,ymm14,ymm3
   d6794:	vpbroadcastd ymm4,DWORD PTR [r9+rax*1+0x20]
   d679b:	vmovdqa ymm1,YMMWORD PTR [rsp+0x840]
   d67a4:	vpmaddubsw ymm4,ymm1,ymm4
   d67a9:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x24]
   d67b0:	vmovdqa ymm1,YMMWORD PTR [rsp+0x820]
   d67b9:	vpmaddubsw ymm5,ymm1,ymm5
   d67be:	vpaddw ymm4,ymm5,ymm4
   d67c2:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x28]
   d67c9:	vmovdqa ymm1,YMMWORD PTR [rsp+0x800]
   d67d2:	vpmaddubsw ymm5,ymm1,ymm5
   d67d7:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x2c]
   d67de:	vmovdqa ymm1,YMMWORD PTR [rsp+0x7e0]
   d67e7:	vpmaddubsw ymm6,ymm1,ymm6
   d67ec:	vpaddw ymm5,ymm6,ymm5
   d67f0:	vpaddw ymm4,ymm5,ymm4
   d67f4:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x30]
   d67fb:	vmovdqa ymm1,YMMWORD PTR [rsp+0x7c0]
   d6804:	vpmaddubsw ymm5,ymm1,ymm5
   d6809:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x34]
   d6810:	vmovdqa ymm1,YMMWORD PTR [rsp+0x7a0]
   d6819:	vpmaddubsw ymm6,ymm1,ymm6
   d681e:	vpaddw ymm5,ymm6,ymm5
   d6822:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x38]
   d6829:	vmovdqa ymm1,YMMWORD PTR [rsp+0x780]
   d6832:	vpmaddubsw ymm6,ymm1,ymm6
   d6837:	vpaddw ymm5,ymm6,ymm5
   d683b:	vpaddw ymm4,ymm5,ymm4
   d683f:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x3c]
   d6846:	vmovdqa ymm1,YMMWORD PTR [rsp+0x760]
   d684f:	vpmaddubsw ymm5,ymm1,ymm5
   d6854:	vpaddw ymm4,ymm5,ymm4
   d6858:	vpmaddwd ymm4,ymm14,ymm4
   d685c:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x40]
   d6863:	vpaddd ymm3,ymm4,ymm3
   d6867:	vmovdqa ymm1,YMMWORD PTR [rsp+0x740]
   d6870:	vpmaddubsw ymm4,ymm1,ymm5
   d6875:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x44]
   d687c:	vmovdqa ymm1,YMMWORD PTR [rsp+0x720]
   d6885:	vpmaddubsw ymm5,ymm1,ymm5
   d688a:	vpaddw ymm4,ymm5,ymm4
   d688e:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x48]
   d6895:	vmovdqa ymm1,YMMWORD PTR [rsp+0x700]
   d689e:	vpmaddubsw ymm5,ymm1,ymm5
   d68a3:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x4c]
   d68aa:	vmovdqa ymm1,YMMWORD PTR [rsp+0x6e0]
   d68b3:	vpmaddubsw ymm6,ymm1,ymm6
   d68b8:	vpaddw ymm5,ymm6,ymm5
   d68bc:	vpaddw ymm4,ymm5,ymm4
   d68c0:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x50]
   d68c7:	vmovdqa ymm1,YMMWORD PTR [rsp+0x6c0]
   d68d0:	vpmaddubsw ymm5,ymm1,ymm5
   d68d5:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x54]
   d68dc:	vmovdqa ymm1,YMMWORD PTR [rsp+0x6a0]
   d68e5:	vpmaddubsw ymm6,ymm1,ymm6
   d68ea:	vpaddw ymm5,ymm6,ymm5
   d68ee:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x58]
   d68f5:	vmovdqa ymm1,YMMWORD PTR [rsp+0x680]
   d68fe:	vpmaddubsw ymm6,ymm1,ymm6
   d6903:	vpaddw ymm5,ymm6,ymm5
   d6907:	vpaddw ymm4,ymm5,ymm4
   d690b:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x5c]
   d6912:	vmovdqa ymm1,YMMWORD PTR [rsp+0x660]
   d691b:	vpmaddubsw ymm5,ymm1,ymm5
   d6920:	vpaddw ymm4,ymm5,ymm4
   d6924:	vpbroadcastd ymm5,DWORD PTR [r9+rax*1+0x60]
   d692b:	vmovdqa ymm1,YMMWORD PTR [rsp+0x640]
   d6934:	vpmaddubsw ymm5,ymm1,ymm5
   d6939:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x64]
   d6940:	vmovdqa ymm1,YMMWORD PTR [rsp+0x620]
   d6949:	vpmaddubsw ymm6,ymm1,ymm6
   d694e:	vpaddw ymm5,ymm6,ymm5
   d6952:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x68]
   d6959:	vmovdqa ymm1,YMMWORD PTR [rsp+0x600]
   d6962:	vpmaddubsw ymm6,ymm1,ymm6
   d6967:	vpbroadcastd ymm7,DWORD PTR [r9+rax*1+0x6c]
   d696e:	vmovdqa ymm1,YMMWORD PTR [rsp+0x5e0]
   d6977:	vpmaddubsw ymm7,ymm1,ymm7
   d697c:	vpaddw ymm6,ymm7,ymm6
   d6980:	vpaddw ymm5,ymm6,ymm5
   d6984:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x70]
   d698b:	vmovdqa ymm1,YMMWORD PTR [rsp+0x5c0]
   d6994:	vpmaddubsw ymm6,ymm1,ymm6
   d6999:	vpbroadcastd ymm7,DWORD PTR [r9+rax*1+0x74]
   d69a0:	vmovdqa ymm1,YMMWORD PTR [rsp+0x5a0]
   d69a9:	vpmaddubsw ymm7,ymm1,ymm7
   d69ae:	vpaddw ymm6,ymm7,ymm6
   d69b2:	vpbroadcastd ymm7,DWORD PTR [r9+rax*1+0x78]
   d69b9:	vmovdqa ymm1,YMMWORD PTR [rsp+0x580]
   d69c2:	vpmaddubsw ymm7,ymm1,ymm7
   d69c7:	vpaddw ymm6,ymm7,ymm6
   d69cb:	vpmaddwd ymm4,ymm14,ymm4
   d69cf:	vpaddw ymm5,ymm6,ymm5
   d69d3:	vpbroadcastd ymm6,DWORD PTR [r9+rax*1+0x7c]
   d69da:	vmovdqa ymm1,YMMWORD PTR [rsp+0x560]
   d69e3:	vpmaddubsw ymm6,ymm1,ymm6
   d69e8:	vpaddw ymm5,ymm6,ymm5
   d69ec:	vpmaddwd ymm5,ymm14,ymm5
   d69f0:	vpaddd ymm4,ymm5,ymm4
   d69f4:	vpaddd ymm6,ymm4,ymm3
   d69f8:	vpxor  xmm3,xmm3,xmm3
   d69fc:	vpxor  xmm5,xmm5,xmm5
   d6a00:	vpxor  xmm4,xmm4,xmm4
   d6a04:	cmp    r10,r13
   d6a07:	je     d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d6a0d:	cmp    r13,r11
   d6a10:	jae    d6ed3 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1de3>
   d6a16:	mov    rax,QWORD PTR [rsp+0x18]
   d6a1b:	movzx  eax,WORD PTR [rax+r13*4]
   d6a20:	lea    rsi,[rax+0x1]
   d6a24:	mov    edx,esi
   d6a26:	shl    edx,0xc
   d6a29:	cmp    rdx,r12
   d6a2c:	ja     d6db0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc0>
   d6a32:	shl    eax,0x5
   d6a35:	shl    esi,0x5
   d6a38:	cmp    rsi,r15
   d6a3b:	ja     d6df4 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d04>
   d6a41:	cmp    rsi,rbx
   d6a44:	ja     d6e0c <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d1c>
   d6a4a:	cmp    r13,0x40
   d6a4e:	jae    d6eeb <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1dfb>
   d6a54:	shl    eax,0x2
   d6a57:	mov    rcx,QWORD PTR [rsp+0x120]
   d6a5f:	vbroadcastss ymm1,DWORD PTR [rcx+rax*1]
   d6a65:	vmulps ymm1,ymm0,ymm1
   d6a69:	mov    rcx,QWORD PTR [rsp+0x28]
   d6a6e:	mov    eax,DWORD PTR [rcx+rax*1]
   d6a71:	shl    eax,0x3
   d6a74:	vmovd  xmm2,eax
   d6a78:	vpbroadcastd ymm2,xmm2
   d6a7d:	vpsubd ymm2,ymm6,ymm2
   d6a81:	vcvtdq2ps ymm2,ymm2
   d6a85:	vmulps ymm1,ymm1,ymm2
   d6a89:	mov    rax,r13
   d6a8c:	shl    rax,0x5
   d6a90:	vaddps ymm1,ymm1,YMMWORD PTR [rsp+rax*1+0x960]
   d6a99:	vmovaps YMMWORD PTR [rsp+rax*1+0x960],ymm1
   d6aa2:	cmp    QWORD PTR [rsp+0x60],0x1
   d6aa8:	je     d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d6aae:	lea    rcx,[r13+0x1]
   d6ab2:	cmp    rcx,r11
   d6ab5:	jae    d6ed0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1de0>
   d6abb:	mov    rax,QWORD PTR [rsp+0x18]
   d6ac0:	movzx  eax,WORD PTR [rax+r13*4+0x4]
   d6ac6:	lea    rsi,[rax+0x1]
   d6aca:	mov    edx,esi
   d6acc:	shl    edx,0xc
   d6acf:	cmp    rdx,r12
   d6ad2:	ja     d6db0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc0>
   d6ad8:	shl    eax,0x5
   d6adb:	shl    esi,0x5
   d6ade:	cmp    rsi,r15
   d6ae1:	ja     d6df4 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d04>
   d6ae7:	cmp    rsi,rbx
   d6aea:	ja     d6e0c <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d1c>
   d6af0:	cmp    r13,0x3f
   d6af4:	je     d6ee8 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1df8>
   d6afa:	shl    eax,0x2
   d6afd:	mov    rdx,QWORD PTR [rsp+0x120]
   d6b05:	vbroadcastss ymm1,DWORD PTR [rdx+rax*1]
   d6b0b:	vmulps ymm1,ymm0,ymm1
   d6b0f:	mov    rdx,QWORD PTR [rsp+0x28]
   d6b14:	mov    eax,DWORD PTR [rdx+rax*1]
   d6b17:	shl    eax,0x3
   d6b1a:	vmovd  xmm2,eax
   d6b1e:	vpbroadcastd ymm2,xmm2
   d6b23:	vpsubd ymm2,ymm4,ymm2
   d6b27:	vcvtdq2ps ymm2,ymm2
   d6b2b:	vmulps ymm1,ymm1,ymm2
   d6b2f:	shl    rcx,0x5
   d6b33:	vaddps ymm1,ymm1,YMMWORD PTR [rsp+rcx*1+0x960]
   d6b3c:	vmovaps YMMWORD PTR [rsp+rcx*1+0x960],ymm1
   d6b45:	cmp    QWORD PTR [rsp+0x60],0x2
   d6b4b:	je     d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d6b51:	lea    rcx,[r13+0x2]
   d6b55:	cmp    rcx,r11
   d6b58:	jae    d6ed0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1de0>
   d6b5e:	mov    rax,QWORD PTR [rsp+0x18]
   d6b63:	movzx  eax,WORD PTR [rax+r13*4+0x8]
   d6b69:	lea    rsi,[rax+0x1]
   d6b6d:	mov    edx,esi
   d6b6f:	shl    edx,0xc
   d6b72:	cmp    rdx,r12
   d6b75:	ja     d6db0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc0>
   d6b7b:	shl    eax,0x5
   d6b7e:	shl    esi,0x5
   d6b81:	cmp    rsi,r15
   d6b84:	ja     d6df4 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d04>
   d6b8a:	cmp    rsi,rbx
   d6b8d:	ja     d6e0c <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d1c>
   d6b93:	cmp    r13,0x3d
   d6b97:	ja     d6ee8 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1df8>
   d6b9d:	shl    eax,0x2
   d6ba0:	mov    rdx,QWORD PTR [rsp+0x120]
   d6ba8:	vbroadcastss ymm1,DWORD PTR [rdx+rax*1]
   d6bae:	vmulps ymm1,ymm0,ymm1
   d6bb2:	mov    rdx,QWORD PTR [rsp+0x28]
   d6bb7:	mov    eax,DWORD PTR [rdx+rax*1]
   d6bba:	shl    eax,0x3
   d6bbd:	vmovd  xmm2,eax
   d6bc1:	vpbroadcastd ymm2,xmm2
   d6bc6:	vpsubd ymm2,ymm5,ymm2
   d6bca:	vcvtdq2ps ymm2,ymm2
   d6bce:	vmulps ymm1,ymm1,ymm2
   d6bd2:	shl    rcx,0x5
   d6bd6:	vaddps ymm1,ymm1,YMMWORD PTR [rsp+rcx*1+0x960]
   d6bdf:	vmovaps YMMWORD PTR [rsp+rcx*1+0x960],ymm1
   d6be8:	cmp    QWORD PTR [rsp+0x60],0x3
   d6bee:	je     d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d6bf4:	lea    rcx,[r13+0x3]
   d6bf8:	cmp    rcx,r11
   d6bfb:	jae    d6ed0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1de0>
   d6c01:	mov    rax,QWORD PTR [rsp+0x18]
   d6c06:	movzx  eax,WORD PTR [rax+r13*4+0xc]
   d6c0c:	lea    rsi,[rax+0x1]
   d6c10:	mov    edx,esi
   d6c12:	shl    edx,0xc
   d6c15:	cmp    rdx,r12
   d6c18:	ja     d6db0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1cc0>
   d6c1e:	shl    eax,0x5
   d6c21:	shl    esi,0x5
   d6c24:	cmp    rsi,r15
   d6c27:	ja     d6df4 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d04>
   d6c2d:	cmp    rsi,rbx
   d6c30:	ja     d6e0c <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d1c>
   d6c36:	cmp    r13,0x3d
   d6c3a:	je     d6ee8 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1df8>
   d6c40:	shl    eax,0x2
   d6c43:	mov    rdx,QWORD PTR [rsp+0x120]
   d6c4b:	vbroadcastss ymm1,DWORD PTR [rdx+rax*1]
   d6c51:	vmulps ymm1,ymm0,ymm1
   d6c55:	mov    rdx,QWORD PTR [rsp+0x28]
   d6c5a:	mov    eax,DWORD PTR [rdx+rax*1]
   d6c5d:	shl    eax,0x3
   d6c60:	vmovd  xmm2,eax
   d6c64:	vpbroadcastd ymm2,xmm2
   d6c69:	vpsubd ymm2,ymm3,ymm2
   d6c6d:	vcvtdq2ps ymm2,ymm2
   d6c71:	vmulps ymm1,ymm1,ymm2
   d6c75:	shl    rcx,0x5
   d6c79:	vaddps ymm1,ymm1,YMMWORD PTR [rsp+rcx*1+0x960]
   d6c82:	vmovaps YMMWORD PTR [rsp+rcx*1+0x960],ymm1
   d6c8b:	jmp    d5d50 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0xc60>
   d6c90:	cmp    rcx,0x20
   d6c94:	jne    d55aa <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x4ba>
   d6c9a:	jmp    d6d58 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c68>
   d6c9f:	cmp    r10,0x4
   d6ca3:	jae    d6ca9 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1bb9>
   d6ca5:	xor    eax,eax
   d6ca7:	jmp    d6d1c <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c2c>
   d6ca9:	mov    ecx,0x60
   d6cae:	xor    eax,eax
   d6cb0:	mov    rdx,QWORD PTR [rsp+0x30]
   d6cb5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   d6cc0:	vmovaps ymm0,YMMWORD PTR [rsp+rcx*1+0x900]
   d6cc9:	vmovups YMMWORD PTR [rdx+rcx*8-0x300],ymm0
   d6cd2:	vmovaps ymm0,YMMWORD PTR [rsp+rcx*1+0x920]
   d6cdb:	vmovups YMMWORD PTR [rdx+rcx*8-0x200],ymm0
   d6ce4:	vmovaps ymm0,YMMWORD PTR [rsp+rcx*1+0x940]
   d6ced:	vmovups YMMWORD PTR [rdx+rcx*8-0x100],ymm0
   d6cf6:	add    rax,0x4
   d6cfa:	vmovaps ymm0,YMMWORD PTR [rsp+rcx*1+0x960]
   d6d03:	vmovups YMMWORD PTR [rdx+rcx*8],ymm0
   d6d08:	sub    rcx,0xffffffffffffff80
   d6d0c:	cmp    rsi,rax
   d6d0f:	jne    d6cc0 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1bd0>
   d6d11:	cmp    QWORD PTR [rsp+0xa0],0x0
   d6d1a:	je     d6d58 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c68>
   d6d1c:	mov    rcx,rax
   d6d1f:	shl    rcx,0x5
   d6d23:	add    rcx,rsp
   d6d26:	add    rcx,0x960
   d6d2d:	shl    rax,0x8
   d6d31:	add    rax,QWORD PTR [rsp+0x30]
   d6d36:	xor    edx,edx
   d6d38:	nop    DWORD PTR [rax+rax*1+0x0]
   d6d40:	vmovaps ymm0,YMMWORD PTR [rcx+rdx*1]
   d6d45:	vmovups YMMWORD PTR [rax+rdx*8],ymm0
   d6d4a:	add    rdx,0x20
   d6d4e:	cmp    QWORD PTR [rsp+0xc8],rdx
   d6d56:	jne    d6d40 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1c50>
   d6d58:	mov    rcx,QWORD PTR [rsp+0x30]
   d6d5d:	add    rcx,0x20
   d6d61:	mov    rax,QWORD PTR [rsp+0x38]
   d6d66:	cmp    rax,0x6
   d6d6a:	lea    rax,[rax+0x2]
   d6d6e:	mov    QWORD PTR [rsp+0x38],rax
   d6d73:	jb     d54dd <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x3ed>
   d6d79:	jmp    d53e7 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x2f7>
   d6d7e:	shl    r8d,0x5
   d6d82:	lea    rcx,[rip+0xa0187]        # 176f10 <__vmlock_lockptr+0x6eb0>
   d6d89:	mov    rdi,r8
   d6d8c:	mov    rdx,r15
   d6d8f:	vzeroupper
   d6d92:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6d97:	shl    r8d,0x5
   d6d9b:	lea    rcx,[rip+0xa0156]        # 176ef8 <__vmlock_lockptr+0x6e98>
   d6da2:	mov    rdi,r8
   d6da5:	mov    rdx,rbx
   d6da8:	vzeroupper
   d6dab:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6db0:	shl    eax,0xc
   d6db3:	jmp    d6e24 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d34>
   d6db5:	shl    r15d,0x5
   d6db9:	lea    rcx,[rip+0xa0150]        # 176f10 <__vmlock_lockptr+0x6eb0>
   d6dc0:	mov    rdi,r15
   d6dc3:	mov    rdx,r11
   d6dc6:	vzeroupper
   d6dc9:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6dce:	shl    r15d,0x5
   d6dd2:	lea    rcx,[rip+0xa011f]        # 176ef8 <__vmlock_lockptr+0x6e98>
   d6dd9:	mov    rdi,r15
   d6ddc:	mov    rdx,rbx
   d6ddf:	vzeroupper
   d6de2:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6de7:	mov    rax,rcx
   d6dea:	jmp    d6e24 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d34>
   d6dec:	mov    rax,rdi
   d6def:	jmp    d6e24 <_RINvNtNtCsuqQadjFOSr_6common2q44avx24gemmKj1000_NCNCNvMs4_NtCsk7rSlHZEixt_13glm_5_3_flash9constructNtBU_10CpuExperts11run_batch_qs2_00NCBM_s_0EBW_+0x1d34>
   d6df1:	shl    eax,0x5
   d6df4:	lea    rcx,[rip+0xa0115]        # 176f10 <__vmlock_lockptr+0x6eb0>
   d6dfb:	mov    rdi,rax
   d6dfe:	mov    rdx,r15
   d6e01:	vzeroupper
   d6e04:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6e09:	shl    eax,0x5
   d6e0c:	lea    rcx,[rip+0xa00e5]        # 176ef8 <__vmlock_lockptr+0x6e98>
   d6e13:	mov    rdi,rax
   d6e16:	mov    rdx,rbx
   d6e19:	vzeroupper
   d6e1c:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6e21:	mov    rax,r8
   d6e24:	lea    rcx,[rip+0xa00fd]        # 176f28 <__vmlock_lockptr+0x6ec8>
   d6e2b:	mov    rdi,rax
   d6e2e:	mov    rsi,rdx
   d6e31:	mov    rdx,r12
   d6e34:	vzeroupper
   d6e37:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6e3c:	lea    rax,[rsp+0x118]
   d6e44:	mov    QWORD PTR [rsp+0x960],rax
   d6e4c:	lea    rax,[rip+0xfffffffffffe630d]        # bd160 <_RNvXsd_NtNtNtCs806nzKq9KaB_4core3fmt3num3impyNtB9_7Display3fmt>
   d6e53:	mov    QWORD PTR [rsp+0x968],rax
   d6e5b:	lea    rcx,[rsp+0x110]
   d6e63:	mov    QWORD PTR [rsp+0x970],rcx
   d6e6b:	mov    QWORD PTR [rsp+0x978],rax
   d6e73:	lea    rdi,[rip+0x7111b]        # 147f95 <_fini+0x5875>
   d6e7a:	lea    rdx,[rip+0x9ffff]        # 176e80 <__vmlock_lockptr+0x6e20>
   d6e81:	lea    rsi,[rsp+0x960]
   d6e89:	call   f9e0 <_RNvNtCs806nzKq9KaB_4core9panicking9panic_fmt>
   d6e8e:	lea    rcx,[rip+0xa0093]        # 176f28 <__vmlock_lockptr+0x6ec8>
   d6e95:	mov    rdi,rax
   d6e98:	mov    rsi,r8
   d6e9b:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6ea0:	lea    rcx,[rip+0xa0069]        # 176f10 <__vmlock_lockptr+0x6eb0>
   d6ea7:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6eac:	lea    rcx,[rip+0xa0045]        # 176ef8 <__vmlock_lockptr+0x6e98>
   d6eb3:	call   fac0 <_RNvNtNtCs806nzKq9KaB_4core5slice5index16slice_index_fail>
   d6eb8:	mov    rcx,r13
   d6ebb:	lea    rdx,[rip+0xa001e]        # 176ee0 <__vmlock_lockptr+0x6e80>
   d6ec2:	mov    rdi,rcx
   d6ec5:	mov    rsi,r11
   d6ec8:	vzeroupper
   d6ecb:	call   fa0b <_RNvNtCs806nzKq9KaB_4core9panicking18panic_bounds_check>
   d6ed0:	mov    r13,rcx
   d6ed3:	lea    rdx,[rip+0xa0006]        # 176ee0 <__vmlock_lockptr+0x6e80>
   d6eda:	mov    rdi,r13
   d6edd:	mov    rsi,r11
   d6ee0:	vzeroupper
   d6ee3:	call   fa0b <_RNvNtCs806nzKq9KaB_4core9panicking18panic_bounds_check>
   d6ee8:	mov    r13,rcx
   d6eeb:	lea    rdx,[rip+0x9ff76]        # 176e68 <__vmlock_lockptr+0x6e08>
   d6ef2:	mov    esi,0x40
   d6ef7:	mov    rdi,r13
   d6efa:	vzeroupper
   d6efd:	call   fa0b <_RNvNtCs806nzKq9KaB_4core9panicking18panic_bounds_check>
   d6f02:	mov    rcx,r13
   d6f05:	lea    rdx,[rip+0x9ffd4]        # 176ee0 <__vmlock_lockptr+0x6e80>
   d6f0c:	mov    rdi,rcx
   d6f0f:	mov    rsi,r10
   d6f12:	vzeroupper
   d6f15:	call   fa0b <_RNvNtCs806nzKq9KaB_4core9panicking18panic_bounds_check>
   d6f1a:	lea    rdx,[rip+0x9ffbf]        # 176ee0 <__vmlock_lockptr+0x6e80>
   d6f21:	xor    edi,edi
   d6f23:	xor    esi,esi
   d6f25:	call   fa0b <_RNvNtCs806nzKq9KaB_4core9panicking18panic_bounds_check>
