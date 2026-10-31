# Kernel patches

Files under `patches/<dir>/` replace the same-named files under `src/linux-7.2.6/drivers/<dir>/`
before every build (`build.sh`). The source tree is unpacked from the pinned tarball and is not
tracked, so this directory is the record of every change to it. Our own changes to these files are
made here, in the tracked copy.

## hwmon/it87.c — the IT8613E Super I/O

metatron's board (HUANANZHI H12D-16D) carries an ITE IT8613E at LPC config port 0x2e with its
Environment Controller at I/O 0xA30: the four fan headers the BIOS's curves drive (two with
tachometers: the CPU coolers), the board thermistors and the voltage rails. Nothing else on the host
can read them. 7.2.6's `it87` does not know the chip; mainline gained it on 2026-08-29 (commits
28e319062, ab003112b, 600a85f556b7 — Benoit Masson / Jerome Tollet, merged by Guenter Roeck). This
file is torvalds master's `drivers/hwmon/it87.c` as of 2026-10-31, which is 7.2.6's file plus that
series (308 lines in 45 hunks, nothing else). It compiles unchanged in our tree.

What the chip looks like on this board, what is documented about it and what we change on top of
mainline: `docs/it8613-superio.md`.

Boot needs `it87.ignore_resource_conflict=1` (`os/release/*.cmdline`): the board's ACPI declares
0xA00–0xA5F as motherboard resources (`PNP0C02`), and the driver refuses the region otherwise.
