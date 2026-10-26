#!/bin/bash
# sibling_cstate.sh <c2|c1|poll>: the idle states the pool cores' SMT siblings (97-119, 121-143, 145-167, 169-191) may
# enter on metatron — c2 the kernel default (acpi_idle: POLL, C1, C2), c1 with C2 disabled, poll with C1 and C2 disabled.
# A runtime switch (sysfs), lost at reboot.
S=$(echo {97..119} {121..143} {145..167} {169..191})
case $1 in c2) d1=0; d2=0;; c1) d1=0; d2=1;; poll) d1=1; d2=1;; *) echo "c2|c1|poll" >&2; exit 1;; esac
$(dirname "$0")/box.sh "for c in $S; do echo $d1 > /sys/devices/system/cpu/cpu\$c/cpuidle/state1/disable; echo $d2 > /sys/devices/system/cpu/cpu\$c/cpuidle/state2/disable; done; cat /sys/devices/system/cpu/cpu152/cpuidle/state1/disable /sys/devices/system/cpu/cpu152/cpuidle/state2/disable | tr '\n' ' '" 8
