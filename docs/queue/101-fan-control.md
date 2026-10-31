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

## Characterization (bringup/box/thermal/load.py + analyze.py; chassis fans at 50 %, CPU coolers and cards on their own curves)

**Pass 1, `warm4k-3min`, 2026-10-31 15:39–15:42:** 12 serial requests of 4k prompt + 256 tokens, 191 s, 48k prompt
tokens. Baseline → peak (°C), time to peak (min), the steepest fall in the minute after the end (°C/min), the cooling
time constant τ (min, log-linear fit over 10 min), time to within 2 °C of baseline (min):

| Series | base | peak | rise | t_peak | fall/min | τ | settle |
|---|---|---|---|---|---|---|---|
| socket 0 Tctl | 51.7 | 66.5 | 14.8 | 2.5 | **−12.4** | **0.6** | 1.2 |
| socket 1 Tctl | 58.8 | 75.6 | 16.9 | 2.9 | **−10.5** | 2.5 | 3.4 |
| card0 c3 junction | 54.4 | 84.0 | 29.6 | 2.9 | **−22.0** | 1.0 | 2.7 |
| card1 a3 junction | 61.0 | 76.0 | 15.0 | 3.0 | −5.0 | 4.3 | 9.2 |
| board thermistor | 30.0 | 39.0 | 9.0 | 3.0 | −2.0 | 6.3 | 9.2 |
| BMC Vcore VR | 51.5 | 68.0 | 16.5 | 3.0 | −6.0 | 2.0 | 4.9 |
| BMC VSoC VR | 53.5 | 71.0 | 17.5 | 3.0 | −7.0 | 2.2 | 5.4 |
| BMC inlet / outlet | 44 / 41.5 | 50 / 50 | 6 / 8.5 | 3.0 | −2.0 | 4.2 / 4.5 | 6.9 / 8.5 |
| socket power (one socket's hwmon) W | 97 | 238 | | | | | |
| cards power W | 113 | 349 | | | | | |
| CPU fan2 rpm | 921 | 1280 | | | | | |
| card0 fan rpm | 0 | 1488 | | | | 7.1 | |

**Pass 2, `decode512-4min`, 16:27–16:31:** 18 requests of ~250 prompt + up to 512 generated tokens, 244 s, 8k
generated. The sockets' work, the cards lighter: Tctl +14.8 / +15.6 (peaks 64.9 / 73.9), VRs +16.4 / +17.0 (68 / 71),
junctions +16.9 / +14.5 (73 / 77), thermistor +4.4; the first-minute falls −9.6 / −8.6 (sockets), −8 / −5 (cards),
−6 / −7 (VRs); τ 3.4–4.2 min for sockets, VRs and card0 (the decay started from a lower, longer plateau than pass 1's,
so the fit reads slower), 6.5 for card1, 8–13 for the ambient group. Socket power (one hwmon) 90 → 191 W, cards
129 → 293 W. Same die physics, a gentler cliff when the cards are not the ones heating.

What it says, and it bears on the goal directly:

- **The dies cool in well under a minute no matter what the fans do.** Socket 0's Tctl has a time constant of ~36 s
  and falls 12 °C in the first minute after a pass, with its fan still at the post-load speed (1280 → 1231 rpm in that
  minute). Card0's junction falls 22 °C in that minute with its fan at 1488 rpm for the next seven. The die's swing is
  the junction-to-heatsink drop, R_jc × P, about 15 °C per 140 W here; that part is physics, and no fan policy slows it.
- **What a fan policy can hold is the slow mass**: the heatsink bases, the VRMs (τ 2 min), the cards' coolers and the
  air around them (τ 4–6 min). The 2026-10-26 week's "14 °C/min" falls were these dies, and they are the same here at
  50 % as at 100 %.
- So the smoothing target is reframed: **hold the heatsinks and the VRM warm at idle, cap the peak, and ramp the air
  down slowly after a pass** so the slow mass descends on the setpoint's slope. The die rides on top of that with its
  fixed R_jc × P step; its amplitude shrinks only through a warmer base at idle (fans slower when nothing runs) and a
  lower peak (fans earlier and harder when a pass starts). Both are what the setpoint scheme does.
- **Card0 took the heat** (+30 °C, fan from zero to 1488 rpm) while card1 moved +15 °C: the upper card in SLOT2 carries
  the attention work in this configuration, or its cooler sits in the warmer air; the per-card zone needs its own
  constants.
- First constants for the daemon's model from this pass: sockets r ≈ 0.11 °C/W, τ_heat ≈ 40 s, τ_cool 36 s (socket 0)
  / 150 s (socket 1); card0 r ≈ 0.25, τ ≈ 60 s; card1 r ≈ 0.12, τ_cool ≈ 260 s; the cards' "loaded" threshold 80 W
  (they idle at 50–55 W with the performance level forced high).

## Open questions

- The BMC's fan-write command (step 3), and whether its daemon reverts a manual duty.
- Which physical header is which: CPU_FAN1/SYS_FAN ↔ pwm2/pwm3; where the three chassis fans plug in today.
- Whether firmware touches the IT8613E's registers at runtime (the ACPI reservation says it may; nothing seen yet).
