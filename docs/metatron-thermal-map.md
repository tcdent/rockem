# metatron's temperature sensors and where they sit

A model of the temperatures inside the enclosure starts with knowing what each sensor touches. This is the
record of that mapping: what is proven, what the data implies, what still needs a hand inside the case.
Started 2026-10-31 (tcdent: map the sensors to physical locations, then model the ranges inside the
enclosure). Data: voria, 3 h of 30 s samples on 2026-10-31 (`bringup/box/thermal/sensors-6h.json`);
registers: `docs/it8613-superio.md`.

## 1. The sensors, by what reads them

| Reader | Sensor | What it is | Where | Evidence |
|---|---|---|---|---|
| k10temp | `Tctl`, `Tccd1..8` per socket | die temperatures, one per CCD | the two SP3 sockets | by definition |
| amdgpu | `edge`, `junction`, `mem`, `fan1` per card | the card's own sensors | the two PCIe slots holding a3:00.0 and c3:00.0 | by definition; which slot is which: §3 |
| IT8613E | `temp1` (TMPIN1) | **socket 0's CPU temperature over SB-TSI** | socket 0 | r = 0.99 with socket 0 `Tctl` at zero lag, 0.89 with socket 1; routed by bank-2 TSS nibble 3 |
| IT8613E | `temp2` (TMPIN2) | **the board's one thermistor** | on the board, in the cards' exhaust path (§2) | `temp3` and 0x2C–0x2E only mirror it (forced-on test 2026-10-31) |
| BMC (IPMI) | `cpu_temp` | socket 0's CPU temperature | socket 0 | r = 0.93 with socket 0 `Tctl`, 0.84 with socket 1; range matches socket 0 |
| BMC | `cpu_vcore_vr_t`, `cpu_vsoc_vr_t` | Vcore and VSoC regulator temperatures, one socket's | a VRM bank | r = 0.96 with each other; 0.87–0.88 with socket 1 `Tctl` against 0.83 with socket 0 — leaning socket 1, not proven |
| BMC | `inlet_temp`, `outlet_temp` | two I2C temperature chips on the BMC's bus | both in the cards' exhaust path (§2); "inlet" is the hotter one | r = 0.96 with each other, inlet ≈ outlet + 1 °C, both r = 0.87 with the cards' junctions |
| amd_hsmp | `power1` per socket | socket power | — | by definition |

Not measured anywhere: the PSU, the DIMMs (no jc42 devices on any SMBus port), socket 1's VRMs if the two VR
sensors are socket 1's, or socket 1's if they are socket 0's.

## 2. What the data says about placement

Three "ambient" sensors — the IT8613E thermistor, BMC inlet, BMC outlet — behave as one group:

- They track the **cards'** junction temperatures (r 0.84–0.88) with a lag of 1.5 to 2.5 minutes, and the CPUs
  only weakly (r ≈ 0.5, at the 5-minute edge of the window). Air heated by the cards reaches them; the CPUs'
  heat does not, or only after mixing.
- Over the 2026-10-26 week the BMC's "inlet" rose from 33 to 56 °C with load. An inlet sensor in fresh air
  does not do that. Either the names are the vendor's board-edge convention rather than this chassis's
  airflow, or the chassis recirculates its exhaust.
- The IT8613E thermistor is 10–12 °C cooler than the BMC pair at every moment (33 against 45 °C), so it sits
  in cooler air, upstream of them or farther from the cards, but in the same stream (same shape, same lag).

The two cards are not equal: a3:00.0 runs ~6 °C hotter at the edge and ~6 °C at the junction than c3:00.0
under the same work, with its fan at the same speed. a3 is downstream of c3 or has less clearance.

The two sockets are not equal: socket 1 idles ~8 °C hotter (58 against 50 °C) and peaks ~4 °C higher. Known
since the bandwidth work (metatron-cpu §2: socket 1 slower, its cooler's torque mattered); a cooler-contact or
airflow difference, not a sensor artifact (k10temp reads the dies directly).

## 2b. The board, from the vendor's 1500 px photograph and the manual (2026-10-31)

Sources: `bringup/box/thermal/dl/board/` (the photo the localai README read part markings from, the
vendor manual `h12d-16d-manual.pdf`, sibling H12D-8D photos). Orientation: rear I/O on the left, slots
bottom-left. Seen unless marked inferred.

- **Sockets.** CPU2 is the left socket, CPU1 the right one; each has a DIMM bank above and below. Linux
  socket 0 = CPU1, socket 1 = CPU2 (the cards sit on NUMA 2 and 3 = socket 1, in CPU2's slots; the Super I/O
  and BMC hang off socket 0's FCH and sit on CPU1's side) — inferred from the NUMA evidence, consistent.
- **Slots.** Seven x16 at the bottom-left, SLOT1 nearest the sockets: CPU2 owns 1, 2, 3, 5, 6; CPU1 owns 4 and 7.
  Our two cards are a3:00.0 (NUMA 3) and c3:00.0 (NUMA 2), both CPU2's, in two of its five slots; which two,
  and the order, is one look inside the case (no firmware slot table on this board).
- **VRMs.** A tall finned heatsink between the sockets (its wide upper part on CPU2's flank, the narrower
  lower part on CPU1's), a heatsinked row of ~8 stages under CPU1's lower DIMM bank, a six-inductor cluster
  under CPU2's lower DIMMs, small clusters on each socket's outer flank. Vendor: "8+6 / 8+6 phases, 60 A
  DrMOS". Inferred assignment: the row under CPU1 = CPU1 Vcore, the heatsink's lower part = CPU1 VSoC, the
  heatsink's upper part = CPU2 Vcore, the cluster = CPU2 VSoC. **Two white 2-pin headers** sit at the foot of
  the central heatsink, one on each socket's side; the product page lists "HS_FAN/MOS 5 V 2-pin", and the
  sibling H12D-8D ships a small fan screwed to that heatsink and cabled to such a header. Ours has none.
- **Fan headers.** CPU_FAN1 top-right corner (CPU1's side), CPU_FAN2 top-left under 12V_PWR2 (CPU2's side):
  these are the IT8613E's two tach-bearing channels, pwm2/pwm3 in some order. SYS_FAN1–3 along the bottom
  edge, SYS_FAN4 on the right edge, SYS_FAN5 top edge near CPU_FAN1: the BMC's. Inferred: the IT8613E's other
  two channels, pwm4 and pwm5 — no tach, the BIOS's "Fan 3/4" curves that start at 32 °C and run flat out by
  127 °C — are the two 2-pin VRM heatsink fan headers, driven as DC fans. That is exactly the shape of a
  tachless VRM fan curve, and it would mean **the host already owns the headers that cool the one component
  that runs to 95 °C**; two 5 V fans on the heatsink is the hardware fix the vendor designed for.
- **Super I/O.** The only LQFP-128 on the board, bottom-edge centre between CLEAR_CMOS/SYS_FAN2 and the 80_LED
  display (inferred from package and position; the marking is not legible). Its one thermistor is an 0603
  part invisible at this resolution; no T_SENSOR header exists anywhere on the board, the manual or the
  product page. The data puts it in the slot bank's exhaust air, the coolest of the ambient group.
- **BMC.** The AST2500 at the bottom edge right of centre, with its DDR4 and its PRIMARY/SECONDARY SPI flash
  pair (`U6`/`U8`); the H12D-8D carries the identical block as a plug-in module, so on this board the "module"
  is laid onto the main board. Consequence for the firmware notes: U6/U8 are the BMC's flash, not a host
  dual-BIOS. The two BMC temperature chips ("inlet", "outlet") are most likely on or beside this block
  (inferred: a transplanted module design, two chips reading within 1 °C of each other, both in the cards'
  exhaust): template names from the OpenBMC configuration, not this chassis's airflow.
- **No public BMC source.** No device tree, firmware image or OpenBMC fork for these boards exists on GitHub
  or Gitee; the sensor names appear nowhere else.

So the ambient sensors all sit along the bottom edge of the board, from centre to right, below the slot
bank. In a tower that edge is the bottom of the case, under the cards; in a rack chassis it is one side.
Which it is here is the chassis question of §3.5.

## 3. What needs a hand or an experiment

1. **Which PCIe slot is which card.** The board doc's slot-to-socket mapping gives slot → root port; a3:00.0 and
   c3:00.0 hang off root ports a0:03.1 and c0:03.1 (NUMA nodes 2 and 3, socket 1). The physical slot order
   decides which card is downstream. One look inside the case.
2. **Which header is which fan.** pwm2/pwm3 are CPU_FAN1 and SYS_FAN in some order; the chassis fans' three BMC
   headers; whether anything sits on the IT8613E's two spare headers. One look.
3. **Differential heating, the clean mapping.** Load one socket's cores for two minutes (bench-power, one
   socket), then the other; load one card, then the other. Whichever VR sensor rises with socket 1 is socket
   1's; the lag and size of the thermistor and inlet/outlet responses to each card place them along the
   airflow. Ten minutes of the box's time, run between serving work; tcdent's call when.
4. **Touch test for the thermistor and the two I2C chips.** A finger or a warm object on a candidate part
   while watching the 10 s series in voria. Candidates from §2b: the two-terminal parts around the LQFP-128
   at the bottom-centre edge for the thermistor; the small 8-pin parts on the BMC block for the pair.
6. **Confirm pwm4/pwm5 are the VRM heatsink fan headers.** Plug a 2-pin 5 V fan into one, set `pwm4`/`pwm5`
   in software mode, watch it. If so, fit two and give the VRMs the cooling the board was designed with.
5. **The chassis itself.** Model, fan positions and directions, board orientation, where the PSU draws from.
   Not derivable from software; needed before any enclosure model means anything.

## 4. Toward the model

With §3 answered the enclosure becomes a few zones with a sensor each: fresh inlet air (none today — a
thermistor on the IT8613E's spare input would be the first thing to add), the CPU/VRM region (dies, VR
sensors, the socket-0 TSI), the card region and its exhaust (junctions, the thermistor, the BMC pair), the
PSU (nothing). The fan-control daemon (`docs/queue/101-fan-control.md`) then has a temperature for each
zone it moves air through, and the ranges come from the week-long voria record rather than from guesses.
