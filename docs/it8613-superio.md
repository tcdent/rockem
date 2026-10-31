# The IT8613E Super I/O on metatron

What the host can see of its own fans, board temperatures and rails without the BMC; what is
documented about the chip, what is guessed, and what the Linux driver has to be for this board.
Research of 2026-10-31 (tcdent: own the driver, extend it, no arbitrary limits on how we talk to the
hardware). The live register dumps are `bringup/box/thermal/sio-ec-dump-2026-10-31.txt`; the sources
gathered (datasheet, driver variants, coreboot, patches) sit under `bringup/box/thermal/dl/`
(reproducible, `CACHEDIR.TAG`).

## 1. The chip as found on the box

Read over `/dev/port` on the serving box, 2026-10-31. Every number below is measured unless marked.

| What | Value |
|---|---|
| LPC config port | 0x2E/0x2F (0x4E answers 0xFF: nothing there) |
| Chip ID / version (SIO 0x20, 0x21, 0x22) | 0x86 0x13, version 8 |
| Environment Controller, LDN 4 | active, base **0xA30** → index **0xA35**, data **0xA36**; second range 0xA20; no IRQ |
| GPIO, LDN 7 | base 0xA10 (simple I/O at 0xA00), not activated |
| CIR, LDN 0xA | base 0x310, IRQ 11, not activated |
| Pin-mux (SIO 0x25–0x2C) | 25=01 26=F3 **27=00 29=00 2A=00** 2B=48 2C=41 |
| ACPI | `PNP0C02` reserves 0xA00–0xA5F, so a hwmon driver needs `it87.ignore_resource_conflict=1` |
| Banks | EC 0x06 bank select bits 6:5; banks 1 and 3 alias bank 0; bank 2 is distinct |

**Fans and PWM.** Four channels are programmed, all in SmartGuardian automatic mode following TMPIN1
(control bytes 0x16, 0x17, 0x1E, 0x1F all 0x80). The curve blocks match the BIOS's four
`IT8613_SMF` knob sets exactly (localai `motherboards/huananzhi-h12d-16d/README.md` §6, `setup-knobs.csv`
rows 51–82), which names the channels:

| Channel | Ctrl / duty / curve base | Live curve (off / start / full °C, start PWM, slope) | BIOS knob block | Tachometer |
|---|---|---|---|---|
| pwm2 | 0x16 / 0x6B / 0x68 | 35 / 40 / 80, 95, 4 PWM per °C | CPU_FAN1 or SYS_FAN | fan2 (0x0E/0x19) 0x02E5 → **~910 rpm** (995 at first boot) |
| pwm3 | 0x17 / 0x73 / 0x70 | same | the other of the two | fan3 (0x0F/0x1A) 0x0320 → **~840 rpm** (915 at first boot) |
| pwm4 | 0x1E / 0x7B / 0x78 | 16 / 32 / 127, 127, 0.5 PWM per °C | Fan 3 or Fan 4 | fan4 (0x80/0x81) reads 0, tach disabled (0x0C bit 4 clear) |
| pwm5 | 0x1F / 0xA3 / 0xA0 | same | the other | fan5 (0x82/0x83) reads 0, tach disabled (0x0C bit 5 clear) |
| pwm1 | 0x15 / 0x63 / 0x60 | factory defaults | — | absent on this chip |

RPM = 1 350 000 / (count × 2), two pulses a revolution (the driver's figure; an earlier note here had
double). Fan main control 0x13 = 0xE0 (fan2, fan3 tachs enabled); fan control
mode 0x14 = 0xC0: PWM **active-high**, base clock 6 MHz → 23.4 kHz PWM. Which physical header is which
is tcdent's to confirm by eye; the two tach-bearing channels are presumably the CPU coolers.

**Temperatures.** 0x29 = 51 °C, 0x2A = 34 °C, 0x2B = 34 °C (0x2C–0x2E also read 34: aliases of TMPIN2,
see §4.1). TMPIN1 tracked socket 0's k10temp Tctl at the time. Temperature enable
0x51 = 0x18 → thermistor bits for TMPIN1 and TMPIN2, nothing for TMPIN3, bits 7:6 clear. Interface
select 0x0A = 0x44 → SMBus external-sensor host on, bits 5:4 = 0. **Bank 2 register 0x1D = 0x03**: the
TMPIN source-select nibble for TMPIN1 is 3, TMPIN2's is 0; 0x1E = 0x00 for TMPIN3. So TMPIN1 is an
external source routed onto the register (the processor over SB-TSI, by every indication), and TMPIN2
and TMPIN3 are the board's own thermistors. Nibble value 3 is not documented anywhere public (only
4 = PECI is); on this board it means AMD TSI, which is the one fact we can add to the record.

**Voltages.** 0x50 = 0x37 enables VIN0, 1, 2, 4, 5; VIN3 and VIN6 are off (their registers read the
same 0xC4, aliases). Raw counts and the 11 mV LSB:

| Input | Reg | Count | at 11 mV | Reading it matches |
|---|---|---|---|---|
| in0 | 0x20 | 0x6C | 1.188 V | Vcore (BMC `vcore_sensor` 1.0–1.5 under load) |
| in1 | 0x21 | 0x76 | 1.298 V | VDDQ (BMC `vddq_sensor` 1.3) |
| in2 | 0x22 | 0xB8 | 2.024 V | a divided rail: 5 V × 0.4 or 12 V × 0.169 — calibrate |
| in4 | 0x24 | 0xB8 | 2.024 V | a divided rail — calibrate |
| in5 | 0x25 | 0xB9 | 2.035 V | a divided rail — calibrate |
| in7 (3VSB, internal, ×2) | 0x27 | 0xAD | 3.81 V | standby rail |
| in8 (VBAT, ×2) | 0x28 | 0x8F | 3.15 V | CR2032 |
| in9 (AVCC3, ×2) | 0x2F | 0x97 | 3.32 V | +3.3 V |

The 11 mV LSB (mainline) against 12 mV (LibreHardwareMonitor) is settled by in1, VBAT and AVCC3: at
12 mV they would read 1.42, 3.43 and 3.62 V. **11 mV is right.**

## 2. Register map: documented against inferred

No IT8613E datasheet is public. coreboot's `src/superio/ite/it8613e` (3mdeb, 2019) was written from
"IT8613E Preliminary Specification V0.3", and its 2026-06 commits (Matt DeVillier, LattePanda Mu) from
the same source; those are the documented facts. The closest public sibling is the IT8728F
Preliminary Specification V0.4.2 (224 pages, `dl/IT8728F-datasheet-rom.by.pdf`), same EC family. The
it87 drivers' IT8613 handling was inferred from user register dumps in April 2017.

| Register | Meaning | Source |
|---|---|---|
| EC 0x00 | configuration; bit 0 start monitoring | datasheet family |
| EC 0x06 | bank select bits 6:5 | coreboot (V0.3) |
| EC 0x0A | interface select: bit 6 SMBus host enable, bits 5:4 00 off / 01 SST slave / 10 PECI / 11 SST host | coreboot (V0.3) |
| EC 0x0C | bit 5 fan5 tach enable, bit 4 fan4 tach enable, **bit 0 AMDTSI error enable** | coreboot (V0.3) |
| EC 0x0D–0x0F, 0x18–0x1A | fan1–3 count lo/hi; limits 0x10–0x12, 0x1B–0x1D | datasheet family |
| EC 0x80/0x81, 0x82/0x83 | fan4, fan5 count; limits 0x84–0x87 | coreboot (V0.3) |
| EC 0x13 | bits 6:4 fan1–3 tach enable; bits 2:0 ON/OFF — **not implemented, SmartGuardian always on** | coreboot `NO_ONOFF` (V0.3) |
| EC 0x14 | bit 7 PWM polarity, bits 6:4 PWM base clock, bit 3 20 % minimum duty | datasheet family |
| EC 0x55 bits 6:4 | PWM base clock for channel 2 | datasheet family |
| **EC 0x15, 0x16, 0x17, 0x1E, 0x1F** | PWM control fan1–5: bit 7 automatic, **bits 5:3 temperature source**, bit 2 closed loop | coreboot (V0.3); fork/mainline agree; LibreHardwareMonitor has pwm4 at 0x7F, wrong |
| EC 0x63, 0x6B, 0x73, 0x7B, 0xA3 | 8-bit duty in software mode | coreboot (V0.3), confirmed on hardware |
| EC 0x60, 0x68, 0x70, 0x78, 0xA0 (+0..+6) | SmartGuardian: off temp, start temp, full temp, start PWM, slope (bits 6:0, 1/8 PWM per °C; bit 7 smoothing), delta/hysteresis (bits 4:0), target zone | datasheet family; coreboot (V0.3) for fan5 at 0xA0 |
| EC 0x20–0x28, 0x2F | VIN0–6, 3VSB, VBAT, AVCC3; limits 0x30+2n/0x31+2n; LSB 11 mV, ×2 for the internal three | LSB: mainline, **verified above** |
| EC 0x29–0x2B | TMPIN1–3; limits 0x40–0x45; offsets 0x56, 0x57, 0x59 (write-enable 0x5C bit 7) | datasheet family |
| EC 0x50 | VIN enable mask | datasheet family |
| EC 0x51 | bits 2:0 diode, bits 5:3 thermistor per TMPIN; **bits 7:6 "reports to" reserved on this chip** | coreboot e778618 (V0.3) |
| **Bank 2 0x1D, 0x1E** | TMPIN source select, a nibble per TMPIN (low = odd); 4 = PECI client 0x30; **3 = AMD TSI on this board** | coreboot (V0.3); our dump |
| EC 0x88–0x8F | external-sensor host (SMBus/PECI/SST): status, address, lengths, command, data, control | datasheet family |
| SIO 0x27 | bit 1 **set** → fan5 tach present; bit 3 → pwm5 is GPIO; bit 6 → pwm3 is GPIO; bit 7 → fan3 is GPIO | it87, inferred 2017; fan5 polarity was wrong until 2025-10 |
| SIO 0x29 | bit 1 → pwm2 is GPIO; bit 2 → fan2 is GPIO; bit 7 → no fan4/pwm4 | it87, inferred |
| SIO 0x2A | bit 0 **clear** → no fan4/pwm4 | it87, inferred — **contradicted by this board** (0x2A = 0, pwm4 carries a BIOS curve) |
| fixed | no fan1/pwm1; no VIN3, VIN6; VIN7 internal; AVCC3 present | it87, inferred; consistent with the dump |

Unknown: the rest of the TSS nibble codes; whether 0x2C–0x2E are TMPIN4–6 (mainline: "only three
temperature inputs are currently known"); what the pin-mux bits really mean on this package.

## 3. The drivers

**Mainline Linux.** IT8613E support landed in torvalds master on 2026-08-29 (28e319062, ab003112b,
600a85f556b7; Benoit Masson / Jerome Tollet, v7 after a Dec 2025 → Aug 2026 review by Guenter Roeck;
`dl/gh/jtollet-v7-*.patch`). Our 7.2.6 predates it. master's `it87.c` against 7.2.6's differs by 308
lines in 45 hunks, every one of them this series, and **compiles unchanged in our 7.2.6 tree** (checked
2026-10-31, builder, no warnings). Its entry: 11 mV ADC, 16-bit fans, five fans and PWMs, PECI, IN7
internal, PWM_FREQ2, AVCC3, the 0x1E/0x1F PWM map, six temperature sources for the PWM map, three
temperature inputs. Tested upstream on an IT8613E at EC 0xA30 revision 12: writing pwm2 = 255 took the
fan from 2636 to 6553 rpm.

**Frank Crawford's fork** (`github.com/frankcrawford/it87`, lineage Roeck → a1wong → hannesha). All
IT8613 work dates from 2017-04-13..19 in Roeck's repo from one user's dump: 11 mV ADC ("officially",
unsourced), the device entry, the pin-mux block, `FEAT_NEW_TEMPMAP`. The fan5 pin test was inverted
for eight years (fixed 2025-10-06). Its own register sheet leaves most IT8613E cells empty. 7671 lines,
of which the Gigabyte H2RAM/MMIO/WMI machinery is dead on this board. It compiles in our tree too, but
it brings nothing mainline lacks for this chip and a lot we would carry for nothing.

**coreboot** owns the only datasheet-derived register definitions (above) and the 2026 finding that
0x51 bits 7:6 are reserved and TSS1 does the routing.

**LibreHardwareMonitor** detects the chip, uses 12 mV and the IT8620E PWM map (pwm4 at 0x7F): wrong on
both counts. Community Windows plugins for Topton/CWWK boards hit "0x7F reverts every write" for the
same reason.

**Boards.** CWWK/Topton N100 ITX, Erying, Aoostar, Protectli, ODROID H4/H5, LattePanda Mu, Beelink,
Minisforum, TerraMaster, UGREEN, and the sibling HUANANZHI H12D-8D (needs the resource-conflict flag,
same as ours).

## 4. What every existing driver gets wrong here, and what ours does

1. **Temperature types.** Fork and mainline type a TMPIN from 0x51 (diode / thermistor bits) and the
   reserved bits 7:6 plus 0x0A for PECI-vs-TSI. On this board that labels TMPIN1 a thermistor when it
   is the CPU over TSI (`temp1_type` reads 4 on the box). Ours reads bank 2's TSS nibbles: nibble ≠ 0
   → external (3 = AMD TSI, 4 = PECI), else the 0x51 bits. **TMPIN3 is not a sensor here:** over 1104
   samples it never differed from TMPIN2, and with its thermistor bit forced on (0x51 = 0x38, 10 s,
   2026-10-31) registers 0x2B and 0x2C still tracked 0x2A step for step. One board thermistor, read on
   TMPIN2; hide temp3 (mainline shows it untyped).
2. **pwm4.** The SIO 0x2A bit 0 rule would hide pwm4 and fan4 on this board while the BIOS drives a
   curve there. Ours decides presence by the EC itself: a channel with a non-default control byte or
   curve is present; the SIO pin bits become a report, not a gate. fan4/fan5 tach presence is 0x0C
   bits 4:5, which the chip already says.
3. **ADC LSB** stays 11 mV (verified). The three divided rails get board-specific scaling once
   calibrated against the BMC's 12 V / 5 V / 3.3 V readings; until then they are raw.
4. **Probe writes.** Mainline writes only three temperature limits and the VIN minimums at probe and
   never touches PWM, duty or curves, so the BIOS state survives the driver loading. Keep that.
5. **pwm4/pwm5's own curves.** Mainline wires the `pwm4/5/6_auto_point*_temp` attributes all to pwm3's
   registers (nr 2 in the attribute table), the IT8620E/IT8628E layout where the extra channels share
   one set of temperature points; `auto_start` and `auto_slope` use the channel's own
   block. On the box pwm5 therefore shows 35/40/80 while its 0xA0 block holds 16/32/127. The IT8613E
   has full blocks at 0x78 and 0xA0 (coreboot, the dump): point the attributes at nr 3 and 4. Mainline's
   visibility arithmetic for pwm4–6 under the newer auto-PWM also hides `pwmN_auto_start` (the
   "no auto_point4" rule lands on the wrong slot of the six-attribute group): fix the index.
6. **PWM frequency.** The shared base clock lives in 0x14, which the drivers expose only through pwm1,
   absent here. Expose it on pwm3+ for this chip.
7. **Fail-safe by design.** SmartGuardian is always on behind software mode: `pwmN_enable = 2` hands a
   channel back to the chip's own curve, which keeps running whatever the host does. The daemon's
   fallback is therefore "write the curve we want as the automatic curve, then run in software mode",
   and any exit, crash or watchdog returns to it with one register write.

## 5. Control tests on the live box (2026-10-31)

**pwm2 (a CPU cooler), through hwmon:** `pwm2_enable=1`, `pwm2=200` → the tach rose 910 → 1290 rpm within
4 s; `pwm2_enable=2` returned the channel to SmartGuardian. **Trap:** the duty register (0x6B) is the
same byte as the curve's start PWM in automatic mode, so after the hand-back `pwm2_auto_start` read 200
and the fan settled at 1530 rpm instead of 910. Writing `pwm2_auto_start=95` restored the BIOS curve.
Any controller that takes a channel into software mode must restore `auto_start` when handing it back,
or write the curve it wants first. All six attributes of the curve are writable live (points, start,
slope, hysteresis, temperature source), so "the fail-safe curve" is programmable, not just the BIOS's.

**The cards' fans, through amdgpu:** `pwm1_enable=1` is accepted (rc 0) but reads back 2: the PMFW
(78.128.0) takes the DisableSmuFeatures(FAN_CONTROL) message and keeps the feature on, so manual duty
is refused ("manual fan speed control should be enabled first"). `pwm1_enable=0` fails outright: the
direct FDO register path finds `FMAX_DUTY100` = 0, so the host has no register-level fan path either.
Manual fan control is impossible on RDNA3 by firmware design: the amdgpu maintainer's words on it
(drm/amd issue 2402, 2023) are that the fan "always remains under the control of the power management
firmware" and only the curve can be tuned, and LACT and CoreCtrl both drive RDNA3 fans through the
overdrive fan table only. That table (`gpu_od/fan_ctrl/*`: `fan_curve`, `fan_minimum_pwm`,
`fan_zero_rpm_enable`, `fan_zero_rpm_stop_temperature`, `fan_target_temperature`, acoustic thresholds)
is absent on metatron for one reason: **our `amdgpu.ppfeaturemask=0xfff73fff` has the overdrive bit
(`PP_OVERDRIVE_MASK` = 0x4000, bit 14) clear** — 0x3fff in the low nibbles clears bits 14 and 15; the
2026-10-15 GFXOFF-off edit of the default 0xfff7bfff never had OD on either. With the bit clear
`smu_set_funcs()` leaves `od_enabled` false and nothing else matters. The fix is a cmdline change and a
reboot: `amdgpu.ppfeaturemask=0xfff77fff` (ours plus 0x4000). An earlier note here blamed the PMFW's
powerplay table (`FeatureCtrlMask` 0); the tables I parsed were the firmware image's, not the combo
table navi31 fetches at boot, and the check never runs with the bit clear — retracted. Zero-RPM mode
has no switch outside that table (kernel 6.13+ `fan_zero_rpm_enable`).

**Done 2026-10-31 15:21 (commit 345b521, lead's reboot): `gpu_od/fan_ctrl/` is up on both cards and every
control works, tested on card0 with the box idle** (`bringup/box/thermal/gpu-od-write-test-2026-10-31.txt`,
script `gpu-od-test.sh`): `fan_minimum_pwm` 15 → 40 + commit took the fan 588 → 1601 rpm in 10 s; a
five-point `fan_curve` (30 °C/20 %, 45/30, 55/60, 70/80, 85/100) + commit ran 2021 rpm at 51 °C junction,
the PMFW interpolating between points (pwm 117–122 of 255 ≈ 46 %); `fan_zero_rpm_enable` 0 accepted;
`r` then `c` on each file returned the card to the PMFW's defaults, and with the junction at 47 °C the
default zero-RPM mode stopped the fan. Ranges: curve hotspot 25–100 °C, speed 15–100 %; minimum PWM
15–100 %; target temperature 25–105 °C (default 81); zero-RPM stop temperature 25–100 °C (default 60);
acoustic limit/target 500–3600 rpm (defaults 3600/1400). `r` needs a following `c` to apply. The PMFW runs
whatever curve is committed, so the cards' fail-safe is the firmware itself; the daemon only moves the
curve, the floor and the zero-RPM switch. `power1_cap` 238–290 W is also writable. The card is a Sapphire 7900 XT (0x1da2:0x471e), VBIOS 113-D70401XT-P11; no
consumer 7900 XT report of overdrive refused by the VBIOS was found. See `docs/queue/101-fan-control.md`.

## 6. The plan

Step 1 is done: kernel #8 (commit ee9b8ff) booted on metatron 2026-10-31 11:26, `it87: Found IT8613E
chip at 0xa30, revision 8`, gates clean, and voria carries the chip's 13 series (`chip="it8613"`) with
no agent change. First-boot readings: fan2 995 / fan3 915 rpm, pwm2/pwm3/pwm5 in automatic mode (pwm4
hidden by the pin-mux rule, as predicted), temp1 54 °C typed thermistor, temp2/temp3 31 °C, rails as in
§1, `intrusion0_alarm` latched at 1. The lead's console capture:
`/tmp/agent-message/rockem.lead/it87-first-boot-2026-10-31.txt`.

1. `os/kernel/patches/hwmon/it87.c` = mainline master's file (the IT8613E series on our 7.2.6 base),
   copied into the untracked tree by `build.sh`; `CONFIG_SENSORS_IT87=y` in `configs/base.config`;
   `it87.ignore_resource_conflict=1` on the cmdline. Our changes on top live in that tracked file.
2. First boot, read everything through hwmon and diff against the dump above; voria-agent picks the
   chip up through `/sys/class/hwmon` with no change, so the new sensors graph beside the rest.
3. Patch 1: pwm4/pwm5's own curve registers and the hidden `auto_start`. Patch 2: TSS-based temperature
   typing. Patch 3: EC-decided channel presence (pwm4). Patch 4: the divided rails' scaling after
   calibration against the BMC (in7 "3VSB" reads 3.8 V, also suspect).
4. Then the daemon (`docs/queue`, to be written): software mode on pwm2–5 and the cards' `pwm1`,
   SmartGuardian curves as the fail-safe, inputs from k10temp, HSMP power, amdgpu and this chip.

Still blind from the host after all this: the VRM temperatures (on the BMC's I2C, not the LPC bus) and
the PSU. A thermistor on a spare TMPIN would cover the PSU; whether the board routes TMPIN3 to a header
or a fixed sensor is not known.
