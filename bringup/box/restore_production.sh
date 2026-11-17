#!/bin/bash
# restore_production.sh: the serving build back on metatron after a bench boot — main's tree, the production code
# object (kernels-aff85), no prow (NODE_FEATURES="--no-default-features"), the short gate workload; the branch that
# was checked out is restored afterwards. A bench boot that hangs the node (PID 1) reboots the box every ~7 min
# until a good image is published (2026-11-02: nine hours of it), so every bench script ends here.
set -u; ROOT=/home/slomp/Work/rockem; cd $ROOT
FROM=$(git branch --show-current)
git switch -q main || { echo "restore: cannot switch to main (uncommitted changes?)"; exit 1; }
[ "$(git branch --show-current)" = main ] || exit 1
bringup/gates/install_elf.sh bundles/tools/kernels-aff85.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash
[ "$(grep -c rockem.step bringup/box/dev.cmdline.serving)" = 0 ] || { echo "restore: main's serving cmdline carries a step key"; exit 1; }
NODE_FEATURES="--no-default-features" bringup/box/deploy.sh restore short 2>&1 | grep -E "pushed|rebooted|node up|api up|cody 18k|hw watch|SCORE|identical|DIFFERS|panick|FAILED" | cut -c1-140
git switch -q "$FROM" && echo "restore: back on $FROM"
