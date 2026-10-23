#!/bin/bash
# lab_boot.sh <label> <cmdline file>: publish the tree's initramfs with the given kernel command line and reboot metatron
# into it, with no workload — a lab boot (an experimental bundle, a model on the CPU, a node built differently). The
# production command line is bringup/box/dev.cmdline.serving: `lab_boot.sh prod bringup/box/dev.cmdline.serving` restores it.
set -u; V=$1; CMD=$2; ROOT=/home/slomp/Work/rockem; LOG=$ROOT/bringup/box/logs
cd $ROOT
cp "$CMD" os/release/dev.cmdline
echo "cmdline  : $(tr ' ' '\n' < os/release/dev.cmdline | grep '^rockem\.' | tr '\n' ' ')"
timeout 500 make initramfs NODE_FEATURES="${NODE_FEATURES---features track-launches}" 2>&1 | tail -n 1 | cut -c1-100
timeout 180 make publish 2>&1 | /usr/bin/grep -E 'pushed|error|missing'
: > $LOG/metatron-console.log
(sleep 1; printf 'echo b > /proc/sysrq-trigger\r\n'; sleep 2) | timeout 8 nc 192.168.88.6 23 >/dev/null 2>&1
echo "rebooted $(date +%H:%M:%S) ($V)"
until timeout 2 bash -c 'echo > /dev/tcp/192.168.88.6/7332' 2>/dev/null; do sleep 5; done; echo "node up $(date +%H:%M:%S)"
/usr/bin/grep -E "ready    :|threads  :|panicked" $LOG/metatron-console.log | sed -E 's/.*\] //' | cut -c1-160 | tail -3
