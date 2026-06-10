# EtherCAT Map

The **EtherCAT Map** window shows the live network topology of slaves attached
to the connected Trio controller. It is useful for confirming that all drives
and I/O modules are present, identified, and in the correct cyclic state.

## Opening the Map

You must be **connected** to a Trio controller. Then click **⚡ EtherCAT Map**
in the left button column. A new window opens showing every slave on the bus.

## What Is Shown

Each slot is drawn as a bus diagram: the controller (master) on the left,
followed by one block per slave. For each slave the map displays:

- **Station address** (above the block) and **position** on the bus.
- **Device name** — read from the device itself via the CoE Identity Object
  (0x1018) and Device Type (0x1000): Trio drives show their model (DX3/DX4…),
  Beckhoff modules are decoded from their product code (e.g. EK1100, EL2002),
  and other devices fall back to their CiA profile (Drive, I/O, Encoder).
- **Vendor** name, resolved from the EtherCAT Technology Group registry.
- **Revision** and online state (green/red bar).
- **Axis** number mapped to the slave (below the block, in blue).
- **Master state** (Init / Pre-Op / Safe-Op / **Op**) on the slot block.

## Device Details

**Click any device block** to open its full identity in the details panel on
the right: vendor (name + ID), product code, revision, serial number, CiA
profile, bus position, station address, mapped axis, drive status and online
state.

### Looking up a device on the internet

- **Search device online** (details panel) opens a web search for the
  selected device in your browser, using the vendor name and decoded product
  so you can quickly find its datasheet or manual.
- **🌐 Update Vendor Names** (toolbar) downloads the official ETG vendor-ID
  registry from [ethercat.org](https://www.ethercat.org/en/vendor_id_list.html)
  so that even uncommon vendors are shown by name instead of a hex ID. The
  list is cached locally (`~/.trioscope/etg_vendors.json`), so it keeps
  working offline afterwards. Without internet access the built-in vendor
  table is used.

## Refreshing

The map is a snapshot. Click **⟳ Scan Network** to re-scan after a topology
change; hot-plug detection is not performed. Reading the device identities
adds a few SDO round-trips per slave, so a scan of a large network can take a
few seconds.

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| Empty map | EtherCAT not started on the controller, or no slaves detected |
| Slaves stuck in **Pre-Op** | Configuration mismatch, missing PDO mapping, or DC sync issue |
| Device shows only a hex product code | Device not in the built-in tables — use **Search device online** |
| Vendor shows as `Unknown (0x…)` | Click **🌐 Update Vendor Names** to fetch the ETG registry |
| Identity fields empty | Slave mailbox not reachable (device offline or still in Init) |
| Window opens but is greyed out | Not connected to a controller — connect first |

For deeper EtherCAT diagnostics use **Motion Perfect** or the controller's
`MECHATROLINK`/`ECAT` BASIC commands.
