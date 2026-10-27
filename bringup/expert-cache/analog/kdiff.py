#!/usr/bin/env python3
"""Every kernel symbol of a code object against a reference object: the disassembly with addresses stripped,
symbol by symbol. Prints the symbols that differ (instruction count each side); exits 1 if any symbol outside
the names in --allow differs. The static gate before a boot (098: the analog's rows variant changed three
production rows kernels through shared code, 2026-10-27)."""
import argparse, glob, os, re, subprocess, sys
a = argparse.ArgumentParser(); a.add_argument("new"); a.add_argument("ref"); a.add_argument("--allow", default="an_chain,an_send")
a.add_argument("--objdump", default=(glob.glob(os.path.expanduser("~/.rustup/toolchains/nightly-*/lib/rustlib/*/bin/llvm-objdump")) + [""])[0])
o = a.parse_args()
def syms(elf):
    out = subprocess.run([o.objdump, "-d", "--mcpu=gfx1100", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    d, cur = {}, None
    for line in out.split("\n"):
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m: cur = m.group(1); d[cur] = []; continue
        if cur and "\t" in line:
            ins = line.split("\t", 1)[1].split("//")[0].strip()
            ins = re.sub(r"\b0x[0-9a-f]{6,}\b", "ADDR", ins)  # branch targets, literals that are addresses
            d[cur].append(ins)
    return d
n, r = syms(o.new), syms(o.ref)
allow = set(o.allow.split(","))
bad = 0
for k in sorted(set(n) | set(r)):
    if n.get(k) != r.get(k):
        tag = "allowed" if k in allow else "CHANGED"
        if k not in allow: bad += 1
        print(f"{tag:8s} {k:24s} {len(r.get(k, []))} -> {len(n.get(k, []))}")
print(f"{len(n)} symbols; {bad} changed outside {sorted(allow)}")
sys.exit(1 if bad else 0)
