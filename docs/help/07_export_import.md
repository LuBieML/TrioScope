# Export & Import

TrioScope can save the entire captured buffer to a CSV file and re-load it
later for offline analysis or sharing.

## Exporting to CSV

1. After (or during) a capture, click **⤓ Export CSV** in the left panel.
2. Pick a filename and location.
3. The file is written with one column per trace plus a `time` column.

### CSV Format

```
time,Trace1_MPOS_axis0,Trace2_FE_axis0,...
0.000000,0.0000,0.0000,...
0.001000,1.2345,-0.0021,...
...
```

- The first column is **time in seconds**, starting from 0.
- Each trace column is named `<trace>_<parameter>_axis<n>`.
- Sample period is derived from the time column.
- Pinned reference traces are **not** exported (only live traces).

## Importing from CSV

1. Click **⤒ Import CSV**.
2. Select a previously exported (or compatible) CSV file.
3. Existing traces are cleared and new traces are created automatically based
   on the column headers.
4. The plot redraws showing the imported data — frozen in time, no live
   capture.

### Compatibility

You can import any CSV that has:

- A first column named `time` (or convertible to numeric time).
- One or more numeric data columns.

Column headers from non-TrioScope sources are imported as-is and become trace
labels. You can rename or recolour them like any normal trace.

## Tips

- **Archive baselines** — export a CSV before any tuning change so you can
  re-import it later as a reference for direct comparison.
- **External analysis** — the CSV format opens cleanly in Excel, MATLAB,
  Python (`pandas.read_csv`), or any other data tool.
- **Long captures** — CSV files can grow to tens of MB for multi-minute
  captures with many traces. Use the Settings dialog to limit window duration
  if you don't need everything.
