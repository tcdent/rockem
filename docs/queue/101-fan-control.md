# 101 — Dynamic fan control from the host

**Status:** Pitched (2026-10-31). The sensor side is live: the IT8613E Super I/O is in the kernel (`docs/it8613-superio.md`,
kernel #8 on metatron since 2026-10-31 11:26) and the CPU coolers' PWM was driven from the host and handed back the same
day. Nothing controls fans dynamically yet; the daemon is unwritten.
**Area:** the box's fans and thermals: the two CPU coolers (Super I/O), the chassis fans (BMC), the cards' fans (PMFW);
the daemon that owns them; the kernel's IPMI path
**Gain:** hardware preservation, not speed: VRM peaks of 92–95 °C under the expert gemm, a 40 °C die swing ~80 times a
day, chassis fans at a fixed duty tcdent sets by hand because the BMC's own profile cannot be trusted

## Direction

> what i'm trying to get us to is a dynamic profile that i can actually trust and so the bmc's dynamic profile i don't
> feel like i can actually trust it i don't know what the cpu coolers are doing i don't think those are managed by the
> bmc and then yeah obviously 100% is falling too fast but 100% is also what's able to maintain a reasonable temperature
> when the system's at full load as well — tcdent, 2026-10-26

> i don't think that that's the right interface necessarily which is why i'm thinking that we want ipmi on the host
> itself and some kind of management system that we develop and deploy ourselves — tcdent, 2026-10-26

> it's really easy for us to write software, especially reverse engineering hardware systems, and so it's not necessary
> that we need someone else to hold our hand to give us what we need. Nor do we need to put any kind of arbitrary
> limitations on the ways in which we can interact with it because we own the driver — tcdent, 2026-10-31

> the bmc is set on the six fans that it can control three of which are connected just at 50% [...] i would like for us
> to come up with a plan to transition to dynamic fan controls based on temperatures — tcdent, 2026-10-31

> we're not extending voria we're building our own ipmi daemon to manage temperatures — tcdent, 2026-10-31

## What the box has (measured 2026-10-26 … 2026-10-31)

Three fan domains, three owners, three control surfaces:

| Domain | Fans | Owner today | Host control surface | State |
|---|---|---|---|---|
| CPU coolers | 2, with tachs (~900 rpm idle) | IT8613E SmartGuardian running the BIOS curve 35/40/80 °C on the CPU temperature | hwmon `pwm2`, `pwm3` (software mode, duty 0–255) and every curve parameter; two more headers `pwm4`, `pwm5` unpopulated | **Works** (test 2026-10-31: 910 → 1290 rpm, handed back) |
| Chassis | 3 of the BMC's 6 headers populated, no tachs read anywhere | BMC (HUANANZHI OpenBMC on the AST2500); tcdent sets a fixed duty by hand, 50 % now, 100 % before | none yet. In-band IPMI exists: KCS at I/O 0xCA2 (SMBIOS type 38), kernel support compiled out | **Open**: the BMC's PWM-write command is unknown |
| Cards | 2 × 7900 XT, own fans, zero-RPM below ~50 °C | the PMFW (78.128.0) | the overdrive fan table `gpu_od/fan_ctrl/*` (curve, minimum PWM, zero-RPM off, target temperature) — absent today only because `amdgpu.ppfeaturemask=0xfff73fff` lacks the overdrive bit 0x4000; manual `pwm1` is impossible on RDNA3 by firmware design | **Live 2026-10-31 15:21**: `0xfff77fff` booted; minimum PWM, curve and zero-RPM switch all verified on card0 |

Inputs the host already has at 100 ms: socket power (HSMP), per-CCD and Tctl (k10temp, both sockets), the cards'
junction/edge/memory temperatures and power, and now the Super I/O's fan tachs, board thermistors and rails. Not on the
host: the VRM temperatures and inlet/outlet (BMC I2C sensors), the PSU (no sensor anywhere).

## The daemon

One process in the OS image, started by init after the node like voria-agent is, owning all three domains. Not voria
(voria observes; see the Direction).

**Policy shape** (from the 2026-10-26 analysis of a week in voria):
- Ramp up on **power**, not temperature: the expert phase reaches 280 W a socket within a second and the dies follow a
  minute later; the VRMs follow the sockets. A controller that sees power leads the transient and clips the VRM peak.
- Hold on temperature, ramp down **rate-limited** (a few percent a minute) so a die cools against its heatsink rather
  than against full airflow. The 14 °C/min falls of 2026-10-26 all happened at 100 % fans.
- A **floor** that is safe for a box that has just crashed at idle. With that floor, a dead daemon is a safe daemon.
- Per domain: CPU coolers from each socket's own temperature and power (socket 1 runs ~11 °C hotter and only the host
  sees it); chassis fans from VR temperature (when reachable), socket+card power, and the PSU estimate (integrated power
  with a minutes-long time constant, until a thermistor exists); the cards from junction temperature if a surface
  appears, otherwise left to the PMFW.

**Fail-safe shape, per domain:**
- CPU coolers: program the curve we want as the SmartGuardian curve (all parameters are writable live), run in software
  mode, and hand back with `pwmN_enable=2` on exit, crash or watchdog. Restore `pwmN_auto_start` on hand-back: the duty
  byte is the start-PWM byte (the 2026-10-31 trap).
- Chassis: whatever the BMC path turns out to be, the last written duty persists if the host dies; so the daemon never
  writes below the floor, and writes high before a pass, so a crash under load leaves fans high.
- Cards: PMFW auto is the fail-safe by construction.

**Observability:** rows into prow (commanded duty per channel, inputs, mode) and nothing else; voria reads the result
through hwmon as it does today. The A/B is the 2026-10-26 queries: VR peaks and time above 85 °C, die amplitude and
cycles a day, steepest fall a minute, duty.

## Steps

1. **Driver patch 1** (hwmon `it87`, ours): pwm4/pwm5 curve registers, the hidden `auto_start`; then TSS-based
   temperature typing and pwm4 presence (docs/it8613-superio.md §4). Each is a kernel build and a deploy.
2. **CPU coolers first.** The daemon with the IT8613E as its only output: software mode on pwm2/pwm3, curve-as-fail-safe,
   rows, watchdog. This alone covers the two fans we can see and control completely, and proves the policy on real data.
3. **Chassis fans, the BMC question.** Two routes, tcdent's call after the first fact is in:
   - *In-band IPMI.* Kernel `IPMI_HANDLER` + `IPMI_DEVICE_INTERFACE` + `IPMI_SI` (KCS at 0xCA2, polled; cap `kipmid_max_busy_us`
     and pin the poller to a housekeeping core). The BMC exports pwm1–6 as IPMI sensors, so the first experiment is Set
     Sensor Reading (NetFn 0x04, cmd 0x30) on a pwm sensor with the BMC's fan daemon in manual mode, which is what the web
     UI does when tcdent sets 50 %. The UI's own request, seen in the browser's developer tools, names the D-Bus object and
     tells us whether an IPMI equivalent exists. The other OpenBMC route is Redfish over the network, which tcdent has set
     aside.
   - *Rewire.* The IT8613E has two unpopulated PWM headers (pwm4, pwm5, the BIOS's "Fan 3/4"). Three chassis fans on two
     headers with one Y-splitter puts every fan the box has under the host's own chip, no BMC in the loop, tachs on two
     of them. Hardware work, tcdent's hands; it makes step 3's software the same as step 2's.
4. **Cards — done 2026-10-31.** `amdgpu.ppfeaturemask=0xfff77fff` is live and `gpu_od/fan_ctrl/` exists on both cards; a
   curve, the minimum PWM and the zero-RPM switch were each written, seen in the tach, and reset. The daemon writes a five-point `fan_curve`, `fan_minimum_pwm` as the
   floor, `fan_zero_rpm_enable=0` so the fans never stop at idle, and commits (`c`). The PMFW keeps running the curve, so
   this domain's fail-safe is built in; the daemon only moves the curve. The other OD knobs (clocks, voltage) stay
   untouched. Caveat from the field: on some 7900 XTs the PMFW overrode zero-RPM settings until a 2026-03 kernel fix;
   7.2.6 has it.
5. **VRM fans.** The board's two 2-pin "HS_FAN/MOS" headers at the foot of the central VRM heatsink are, by
   every sign, the IT8613E's pwm4/pwm5 (docs/metatron-thermal-map.md §2b). Two 5 V fans there, driven by the
   daemon from the VR temperatures (BMC) or socket power, attack the 92–95 °C peaks at their source. Hardware:
   two fans and a look; software: the same channel code as step 2.
6. **PSU.** A thermistor on a spare TMPIN of the IT8613E if the board routes one to a header; else the power-integral
   estimate stands.
7. **Measure.** Two weeks A/B in voria with the queries above; then the IPMI question for the BMC's own sensors (VR
   temperatures) if step 3 took the in-band route.

## Open questions

- The BMC's fan-write command (step 3), and whether its daemon reverts a manual duty.
- Which physical header is which: CPU_FAN1/SYS_FAN ↔ pwm2/pwm3; where the three chassis fans plug in today.
- Whether firmware touches the IT8613E's registers at runtime (the ACPI reservation says it may; nothing seen yet).
