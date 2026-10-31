# metatron's heat map, in two dimensions

The enclosure as a plane the air crosses front to back, with every heat source, sensor and fan placed on it.
Started 2026-10-31 (tcdent: a two-dimensional understanding of the heat inside the case, zones for the three
front fans, curves that hold temperatures instead of chasing them). Sensor identities and the board's
layout: `docs/metatron-thermal-map.md`; the chip and the cards: `docs/it8613-superio.md`; the plan:
`docs/queue/101-fan-control.md`.

## 1. Frame

**The chassis (tcdent, 2026-10-31): a Sliger CX4150a** — 4U, 15" deep, the board lying flat; three 120 mm
Noctuas side by side across the front, one 120 mm rear exhaust position; rated for motherboards to 10.5"
deep, so this 12"-deep E-ATX board overhangs toward the front fans (the reason for the CX4170a, 17" deep, in
the mail). Both CPUs under Noctua 120 mm tower coolers. Two Radeon 7900 XT (324 mm is the chassis's card
limit). A Corsair ATX power supply with its own 140 mm fan, which draws from inside the case, sits at the
rear **in socket 1's corner**: tcdent sees it bleed warm air into that socket's region whenever the case
lacks positive pressure, on top of socket 1 already breathing socket 0's exhaust.

So the plane is horizontal. Air enters across the front edge of the board (the SATA/MCIO/M.2 edge) and
leaves at the rear I/O edge; the three front fans divide the board's *width*, not a height. In board-photo
terms (rear I/O on the left) the air crosses the photo **right to left**; the photo's "up" is the chassis's
side where the PSU sits. Coordinates below: **x** = distance from the front of the board (0 at the front
edge, 330 mm at the rear I/O); **y** = position across the chassis from the slots' edge (0) to the ATX-power
edge (305 mm), the PSU side. From the vendor's 1500 px photo at ≈0.25 mm/px; ±10 mm.

## 2. Everything that heats, senses or moves air

| Item | x from front (mm) | y from bottom (mm) | Heat / reading at idle → peak | Notes |
|---|---|---|---|---|
| **CPU1 = socket 0** | 70–180 | 150–225 | 75 → 280 W; Tctl 50 → 75 °C | first socket the front air meets; k10temp; also the SIO's temp1 and the BMC's cpu_temp |
| CPU1 upper DIMM bank | 70–185 | 230–270 | — | no sensor (no jc42) |
| CPU1 lower DIMM bank | 70–185 | 95–135 | — | no sensor |
| CPU1 Vcore VRM row (inferred) | 70–165 | 65–80 | — | heatsinked, under CPU1's lower DIMMs; sensed only if the BMC's `vr_t` pair is socket 0's |
| **Central VRM heatsink** | 175–200 | 85–220 | 92–95 °C peaks (BMC `cpu_vcore_vr_t`, `cpu_vsoc_vr_t`, one socket's) | the finned tower between the sockets; **two 2-pin fan headers at its foot, unpopulated** (pwm4/pwm5, inferred) |
| **CPU2 = socket 1** | 200–315 | 175–250 | 80 → 280 W; Tctl 58 → 87 °C | downstream of CPU1 and the VRM tower, ~8 °C hotter at idle; k10temp only — no other sensor sees it |
| CPU2 DIMM banks | 195–315 | 255–290 and 125–160 | — | no sensor |
| CPU2 VSoC cluster (inferred) | 180–205 | 60–105 | — | six inductors under CPU2's lower DIMMs |
| **Card c3:00.0 (card0), SLOT2** | ~45–255 (card length) | ~85–100 slot, cooler to ~45 | 18 → 144 W; junction 44 → 81 °C | NUMA 2; upper of the two cards; tcdent: SLOT2, SLOT1 holds the NIC |
| **Card a3:00.0 (card1), SLOT5** | ~45–255 | ~30–45 slot, cooler to ~0 | 16 → 134 W; junction 47 → 81 °C, runs ~6 °C hotter than card0 | NUMA 3; lower card; tcdent: "4 or 5" → SLOT5 (SLOT4 is CPU1's; a3 is socket 1's) |
| NIC (82599 2×10G), SLOT1 | ~45–180 | ~110 | ~10 W | NUMA 2, root e0; above card0 |
| **Super I/O IT8613E** | 165–180 | 10–25 | `temp2`: 28 → 34 °C | bottom-centre edge; its thermistor within a few cm (unlocated) |
| **BMC block (AST2500)** | 95–120 | 10–30 | `inlet`/`outlet`: 33 → 56 °C | bottom edge, front of centre; the two I2C temperature chips on or beside it (inferred) |
| CPU_FAN1 header | 5 | 300 | drives the socket-0 cooler (pwm2 or pwm3) | top-front corner |
| CPU_FAN2 header | 325 | 300 | drives the socket-1 cooler (the other) | top-rear corner, under 12V_PWR2 |
| SYS_FAN5 | 40 | 300 | BMC | top edge, front |
| SYS_FAN4 | 5 | 95 | BMC | front edge, low |
| SYS_FAN1, SYS_FAN2, SYS_FAN3 | 150, 130, 55 | 5 | BMC | bottom edge |
| **PSU** (Corsair, 140 mm fan) | 330+ (rear) | ~305+ (beyond the ATX-power edge, socket 1's corner) | no sensor; draws from inside the case; bleeds into socket 1's region without positive pressure (tcdent) | the fourth exhaust nobody controls |
| Rear exhaust fan position | 330+ | ~150 (behind the sockets) | — | fitted or not: a look; BMC header if fitted |

Three of the BMC's six headers carry the three front fans (tcdent); which three is a look. Front fans
mount on the case, not the board: their y positions across the chassis width are what define the zones.
With the board overhanging the chassis's motherboard depth, the front edge of the board sits close to or
against the fan plane: the cards' noses and the MCIO/SATA block are the first things the fans hit.

## 3. The plane at idle and at peak

Air enters at the front (x = 0) and leaves at the rear (x = 330). Reading the table as a picture, idle on the
left of each cell, peak on the right; "—" is a region with no sensor:

```
 y (mm)   front ──────────────────────────── air ────────────────────────────▶ rear
 300  │  CPU_FAN1 ·  SYS_FAN5   ATX/12V power            12V_PWR2 · CPU_FAN2
 260  │  [CPU1 upper DIMMs —]                [CPU2 upper DIMMs —]
 200  │  [ CPU1 / socket 0   50→75 ]  ▐VRM▌   [ CPU2 / socket 1   58→87 ]
 150  │                               ▐tower▌
 110  │  [CPU1 lower DIMMs —]  NIC    ▐92-95▌  [CPU2 lower DIMMs —]
  90  │  [CPU1 Vcore row —] SLOT2: card0  44→81 junction  (cooler hangs below)
  40  │                     SLOT5: card1  47→81 junction  (+6 °C vs card0)
  10  │  SYS_FAN3   BMC block 33→56 "inlet/outlet"   SIO + thermistor 28→34   SYS_FAN2 SYS_FAN1
      └──────────────────────────────────────────────────────────────────────────
        x = 0                                                              x = 330
```

What the picture already says (read it with y across the chassis: the slots' side at the bottom of the
drawing, the PSU side at the top):

- **The hottest things are in the middle strip**: the VRM tower and the two sockets at y 85–250.
  Socket 1 sits in air that has already crossed socket 0 and the tower, reads 8 °C hotter at idle for it,
  and has the PSU's recirculation on its far side.
- **The cards occupy the slots' strip**, and everything that senses "ambient" is in that strip too, along
  the slots' edge. The ambient sensors measure the card region, not the case.
- **The PSU-side strip has no sensor and little heat of its own**: DIMMs and power connectors — but it is
  where the PSU's warm air arrives if pressure is low.
- Nothing measures the inlet air itself. The first sensor the air meets is socket 0's die.

## 4. Zones for three front fans

Three 120 mm fans side by side across a chassis about 430 mm wide cover the board's 305 mm width and some
margin; each fan's 120 mm lane maps onto the board's strips:

| Fan lane | Covers (y) | Heat it carries | Sensor to hold | Also |
|---|---|---|---|---|
| PSU side | ~200–305 | the sockets' upper halves, upper DIMMs, the power connectors; the lane the PSU's bleed enters from the rear | the two Tctls | this lane's pressure is what keeps the PSU's air out of socket 1 |
| Middle | ~85–200 | the VRM tower, the sockets' lower halves, lower DIMMs, the NIC | `cpu_vcore_vr_t` / `cpu_vsoc_vr_t` (BMC) and the Tctls | the two CPU coolers draw from this lane |
| Slots side | ~0–85 | the two cards' coolers and their exhaust, the SIO/BMC region | the junctions (amdgpu), with `temp2` and the BMC pair as the exhaust check | the cards' coolers dump into the case, not out the back |

**Pressure is a zone of its own.** Exhaust is the rear 120 mm position (if fitted), the PSU's 140 mm fan
(uncontrolled), and the rear I/O gaps; the cards' coolers recirculate inside. Positive pressure means the
three intakes move more air than those exhausts. The PSU bleed tcdent sees is the sign it is not always so,
which argues for a floor on the three front fans that no setpoint may undercut, and for the PSU-side lane
to carry the highest floor. In the CX4170a the same geometry gets 2" more depth between the board's front
edge and the fans, which is airflow the cards' noses currently block.

Plus the sources that have their own fans: the two CPU coolers (the SIO's pwm2/pwm3, each from its socket's
Tctl and power), the two cards (the PMFW's curve, shaped through the overdrive table once it exists), and
the VRM tower (two 2-pin fans on pwm4/pwm5 once fitted, from the VR temperatures).

Which BMC header each front fan is on, and whether the three sit top/middle/bottom or side by side, is the
look that turns this table from a proposal into a configuration.

## 5. The curves: hold, don't chase

The goal (tcdent, 2026-10-31): fewer and shallower heating/cooling cycles — idle a little warmer, a
consistent temperature at peak, a steady descent after it. In control terms every zone tracks a
**setpoint that moves slowly**, rather than a fan curve that reacts to temperature:

- **Idle setpoint** above today's idle (socket 0 50 °C, socket 1 58 °C, cards 45–50 °C): the fans slow until the
  zone sits there. Warmer idle, smaller swing.
- **Peak setpoint** below today's peaks (socket 1 87 °C, VRs 95 °C, junctions 81 °C): on a power rise the
  fans go up ahead of the temperature (power leads by about a minute), and hold the zone at the setpoint
  rather than letting it run to wherever 100 % airflow lands.
- **The descent is the setpoint's, not the fan's**: when the load ends, the setpoint ramps from peak to idle
  at a bounded rate (a degree a minute, to be tuned), and the fans do whatever holds the zone on that ramp.
  Today's 14 °C/min falls become a controlled slope. The fans end up *slower* right after a pass than
  during it, which is the opposite of a reactive curve and the whole point.
- Each zone's loop is a plain PI on its sensor against the setpoint with the fan duty clamped between a
  floor (safe for a dead daemon) and 100 %, plus the feed-forward from power. The SmartGuardian and PMFW
  curves underneath stay programmed as the fail-safe.

What the loops need that is not yet in hand: the chassis geometry and fan headers (one look), the VRM
fans (two parts), the cards' overdrive table (one reboot, in flight), and the differential-heating run that
tells the VR sensors' socket and the ambient sensors' response to each card.
