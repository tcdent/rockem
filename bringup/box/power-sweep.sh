# power-sweep.sh: bench-power over every kind of work at 1 / 23 / 46 / 92 cores (socket 0 first), into /tmp/power.out
: > /tmp/power.out
for m in idle chain vmadd l1 l2 l3 dram gemm gemm-dram; do
  for c in 1 1-23 1-23,25-47 1-23,25-47,49-71,73-95; do
    /bundle/tools/bench-power --mode $m --cpus $c --secs 5 2>&1 | tr "\n" " " >> /tmp/power.out
    echo >> /tmp/power.out
    sleep 2
  done
done
echo done >> /tmp/power.out
