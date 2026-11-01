#!/bin/sh
# fan-map.sh: which IT8613E PWM channel cools which socket. At idle, each channel is run at duty 220 in
# software mode for 90 s while both sockets' Tctl are sampled; the socket whose die drops is that
# channel's. Each channel is handed back to SmartGuardian with its start PWM restored. Runs on the box.
H=/sys/class/hwmon/hwmon2
T0=/sys/class/hwmon/hwmon3/temp1_input   # k10temp 00:18.3 = socket 0 (Tctl is temp1 on both)
T1=/sys/class/hwmon/hwmon4/temp1_input   # k10temp 00:19.3 = socket 1
t(){ echo $(( $(cat $1) / 100 )); }       # tenths of a degree
avg(){ s=0; n=0; for i in 1 2 3 4; do s=$((s + $(t $1))); n=$((n+1)); sleep 3; done; echo $((s / n)); }
for c in 2 3; do
  start=$(cat $H/pwm${c}_auto_start)
  echo "== pwm$c: baseline (12 s)"; b0=$(avg $T0); b1=$(avg $T1)
  echo "   s0=$b0 s1=$b1 (tenths C) fan2=$(cat $H/fan2_input) fan3=$(cat $H/fan3_input)"
  echo 1 > $H/pwm${c}_enable; echo 220 > $H/pwm$c
  for k in 1 2 3 4 5 6; do sleep 15; echo "   +$((k*15))s s0=$(t $T0) s1=$(t $T1) fan2=$(cat $H/fan2_input) fan3=$(cat $H/fan3_input)"; done
  e0=$(avg $T0); e1=$(avg $T1)
  echo 2 > $H/pwm${c}_enable; echo $start > $H/pwm${c}_auto_start
  echo "   end: s0=$e0 s1=$e1  drop s0=$((b0 - e0)) s1=$((b1 - e1)) tenths; handed back (enable=$(cat $H/pwm${c}_enable) start=$(cat $H/pwm${c}_auto_start))"
  d0=$((b0 - e0)); d1=$((b1 - e1))
  if [ $((d0 - d1)) -gt 10 ]; then echo "   => pwm$c cools SOCKET 0"; elif [ $((d1 - d0)) -gt 10 ]; then echo "   => pwm$c cools SOCKET 1"; else echo "   => pwm$c: no clear difference (${d0} vs ${d1})"; fi
  echo "   settling 60 s"; sleep 60
done
