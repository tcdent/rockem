#!/usr/bin/env python3
"""The code object's static report: a crude linter run after every GPU build.

  python3 bringup/kernel_report.py <kernels.elf> [--show]

Reads each kernel's registers, scratch, LDS and a few instruction counts from the
ELF (llvm-readobj notes, llvm-objdump) and keeps the last report beside the ELF as
`<elf>.report.json`. Prints nothing when nothing changed and every kernel is within
bounds; otherwise one line per kernel that moved, and what moved. Fails the build on
scratch (the node's loader refuses it anyway; better here), on more registers than the
occupancy bound, or on a barrier count that changed (the run records the new counts, so a second run accepts an intended change) — the duplicated-barrier bug of
2026-09-19 would have printed here. `--show` prints the whole table regardless."""
import argparse, glob, json, os, re, subprocess, sys

VGPR_MAX = 192          # past this a SIMD holds fewer than 8 waves
# Kernels whose workgroups must all be resident at once (device barriers inside): (workgroups, waves each). The 7900 XT
# has 84 CUs, 168 SIMD32s of 1536 VGPRs a lane; a kernel past the registers its grid allows hangs, it does not slow
# down (hc_pre at 189 VGPRs ran 168 of its 192 workgroups and its barriers waited forever, 2026-10-20).
SIMDS, VGPRS_PER_SIMD, VGPR_GRANULE = 168, 1536, 8
GRID_RESIDENT = {"hc_pre": (192, 8)}
def resident_vgpr_max(wgs, waves):
    per_simd = -(-wgs * waves // SIMDS)
    return VGPRS_PER_SIMD // per_simd // VGPR_GRANULE * VGPR_GRANULE
INSTR_JUMP = 0.25       # fraction of change in a kernel's instruction count worth a line

ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--show", action="store_true"); a = ap.parse_args()
tool = glob.glob(os.path.expanduser("~/.rustup/toolchains/nightly-*/lib/rustlib/*/bin/"))
if not tool:
    sys.exit("no nightly llvm tools under ~/.rustup")
readobj, objdump = os.path.join(tool[0], "llvm-readobj"), os.path.join(tool[0], "llvm-objdump")

notes = subprocess.run([readobj, "--notes", a.elf], capture_output=True, text=True, check=True).stdout
kernels = {}
name = None
for line in notes.splitlines():
    m = re.match(r"\s*\.name:\s+(\S+)", line)
    if m:
        name = m.group(1); kernels[name] = {}
        continue
    for key, field in (("vgpr", ".vgpr_count"), ("sgpr", ".sgpr_count"), ("scratch", ".private_segment_fixed_size"), ("lds", ".group_segment_fixed_size")):
        m = re.match(rf"\s*{re.escape(field)}:\s+(\d+)", line)
        if m and name:
            kernels[name][key] = int(m.group(1))
dis = subprocess.run([objdump, "-d", "--mcpu=gfx1100", a.elf], capture_output=True, text=True, check=True).stdout
cur = None
for line in dis.splitlines():
    m = re.match(r"^[0-9a-f]+ <(\w+)>:", line)
    if m:
        cur = m.group(1) if m.group(1) in kernels else None
        if cur:
            kernels[cur].update(instr=0, barriers=0, loads=0, lds_ops=0)
        continue
    if cur and line.startswith(("\t", " ")):
        k = kernels[cur]; k["instr"] += 1
        if "s_barrier" in line: k["barriers"] += 1
        if "global_load" in line or "buffer_load" in line: k["loads"] += 1
        if "ds_load" in line or "ds_store" in line: k["lds_ops"] += 1

path = a.elf + ".report.json"
prev = json.load(open(path)) if os.path.exists(path) else {}
lines, fail = [], []
for n in sorted(kernels):
    k, p = kernels[n], prev.get(n)
    row = f"{n:<14} vgpr {k.get('vgpr',0):>3}  scratch {k.get('scratch',0):>4}  lds {k.get('lds',0):>5}  instr {k.get('instr',0):>5}  barriers {k.get('barriers',0):>2}  loads {k.get('loads',0):>3}  lds_ops {k.get('lds_ops',0):>3}"
    why = []
    if k.get("scratch", 0):
        why.append("SCRATCH"); fail.append(n)
    if k.get("vgpr", 0) > VGPR_MAX:
        why.append(f"vgpr > {VGPR_MAX}"); fail.append(n)
    if n in GRID_RESIDENT and k.get("vgpr", 0) > resident_vgpr_max(*GRID_RESIDENT[n]):
        why.append(f"vgpr > {resident_vgpr_max(*GRID_RESIDENT[n])}: its grid no longer fits the card at once (its barriers would hang)"); fail.append(n)
    if p is None:
        why.append("new")
    else:
        if p.get("barriers") != k.get("barriers"):
            why.append(f"barriers {p.get('barriers')} -> {k.get('barriers')}"); fail.append(n)
        if p.get("instr") and abs(k["instr"] - p["instr"]) > INSTR_JUMP * p["instr"]:
            why.append(f"instr {p['instr']} -> {k['instr']}")
        if p.get("vgpr") is not None and abs(k.get("vgpr", 0) - p["vgpr"]) >= 16:
            why.append(f"vgpr {p['vgpr']} -> {k.get('vgpr')}")
        if p.get("scratch", 0) != k.get("scratch", 0):
            why.append(f"scratch {p.get('scratch')} -> {k.get('scratch')}")
    if why or a.show:
        lines.append(row + ("   " + ", ".join(why) if why else ""))
gone = sorted(set(prev) - set(kernels))
if lines or gone:
    print(f"kernel report, {len(kernels)} kernels ({os.path.basename(a.elf)}):")
    for l in lines: print("  " + l)
    for g in gone: print(f"  {g:<14} gone")
else:
    print(f"kernel report: {len(kernels)} kernels, no scratch, nothing moved")
json.dump(kernels, open(path, "w"), indent=1)
if fail:
    sys.exit(f"kernel report: {', '.join(sorted(set(fail)))} — barrier counts must not change without a reason, and scratch or registers past {VGPR_MAX} are never shipped")
