# EtherCAT Map

The **EtherCAT Map** window shows the live network topology of slaves attached
to the connected Trio controller. It is useful for confirming that all drives
and I/O modules are present, identified, and in the correct cyclic state.

## Opening the Map

You must be **connected** to a Trio controller. Then click **⚡ EtherCAT Map**
in the left button column. A new window opens listing every slave on the bus.

## What Is Shown

For each slave the map displays:

- **Position** on the EtherCAT bus (slave index).
- **Vendor** and **Product** identifiers.
- **Revision** and **Serial number**.
- **AL state** (Init / Pre-Op / Safe-Op / **Op**).
- **Alias address**, if configured.
- **Mailbox protocols** supported (CoE, FoE, EoE, …).

If the slave is recognised as a Trio DX-series drive, additional
drive-specific information may be shown (e.g. firmware version).

## Refreshing

The map is a snapshot taken at the moment the window is opened. Re-open the
window to refresh after a topology change. Hot-plug detection is not
performed.

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| Empty map | EtherCAT not started on the controller, or no slaves detected |
| Slaves stuck in **Pre-Op** | Configuration mismatch, missing PDO mapping, or DC sync issue |
| Wrong vendor / product IDs | Slave needs configuration; check the EtherCAT XML in Motion Perfect |
| Window opens but is greyed out | Not connected to a controller — connect first |

For deeper EtherCAT diagnostics use **Motion Perfect** or the controller's
`MECHATROLINK`/`ECAT` BASIC commands.
