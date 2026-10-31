#!/bin/sh
# gpu-od-test.sh CARD: on the box. Raise the minimum PWM, apply a curve, switch zero-RPM off, reset each; print the tach.
C=${1:-0}; D=/sys/class/drm/card$C/device; F=$D/gpu_od/fan_ctrl; H=$(dirname $(ls $D/hwmon/hwmon*/fan1_input | head -1))
r(){ echo "$(date +%T) fan=$(cat $H/fan1_input) pwm=$(cat $H/pwm1) junction=$(cat $H/temp2_input) min=$(sed -n 2p $F/fan_minimum_pwm) zr=$(sed -n 2p $F/fan_zero_rpm_enable)"; }
echo "-- card$C before"; r
echo "-- minimum pwm 40 + commit"; echo 40 > $F/fan_minimum_pwm; echo "write rc=$?"; echo c > $F/fan_minimum_pwm; echo "commit rc=$?"
sleep 5; r; sleep 5; r
echo "-- reset minimum"; echo r > $F/fan_minimum_pwm; echo "rc=$?"; sleep 6; r
echo "-- curve 30/20 45/30 55/60 70/80 85/100 + commit"
for p in "0 30 20" "1 45 30" "2 55 60" "3 70 80" "4 85 100"; do echo "$p" > $F/fan_curve || echo "point $p rc=$?"; done
echo c > $F/fan_curve; echo "commit rc=$?"; sed -n 2,6p $F/fan_curve | tr "\n" " "; echo
sleep 6; r; sleep 6; r
echo "-- zero rpm off + commit"; echo 0 > $F/fan_zero_rpm_enable; echo c > $F/fan_zero_rpm_enable; echo "rc=$? zr=$(sed -n 2p $F/fan_zero_rpm_enable)"; sleep 3; r
echo "-- reset curve and zero rpm"; echo r > $F/fan_curve; echo r > $F/fan_zero_rpm_enable; echo "rc=$?"; sed -n 2,6p $F/fan_curve | tr "\n" " "; echo
sleep 8; r; sleep 6; r
dmesg | tail -5 | grep -i amdgpu
