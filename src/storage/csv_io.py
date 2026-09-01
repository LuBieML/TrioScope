import csv
import re
from typing import Dict, List, Sequence, Tuple
import numpy as np

# Column names like "MPOS(0)" -> param="MPOS", axis=0
_PARAM_PATTERN = re.compile(r'^(.+)\((\d+)\)$')
_SEGMENT_COLUMN = "__TRIOSCOPE_SEGMENT__"


class CSVStorage:
    @staticmethod
    def export_data(path: str, time_data: np.ndarray,
                    params_data: Dict[str, np.ndarray],
                    segment_breaks: Sequence[int] | None = None) -> None:
        """Export capture data, including continuous-capture boundaries."""
        param_names = list(params_data.keys())
        if _SEGMENT_COLUMN in param_names:
            raise ValueError(f"Reserved CSV column name: {_SEGMENT_COLUMN}")
        n = len(time_data)
        breaks = sorted({int(b) for b in (segment_breaks or ())
                         if 0 < int(b) < n})
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Time']
            # writerows over zipped columns avoids per-row list building in
            # Python — noticeably faster for 100k+ sample captures.
            columns = [np.round(time_data, 6)]
            if breaks:
                segment_ids = np.zeros(n, dtype=np.int64)
                for boundary in breaks:
                    segment_ids[boundary:] += 1
                header.append(_SEGMENT_COLUMN)
                columns.append(segment_ids)
            header.extend(param_names)
            columns.extend(params_data[p] for p in param_names)
            writer.writerow(header)
            writer.writerows(zip(*columns))

    @staticmethod
    def import_data(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Tuple[str, int]]]:
        """
        Import time and parameter data from a CSV file.
        
        Returns:
            Tuple of (time_array, params_dict, list_of_trace_tuples)
            where list_of_trace_tuples is a list of (param_name, axis) parsed from headers.
        """
        time_arr, params, traces, _ = CSVStorage.import_data_with_metadata(path)
        return time_arr, params, traces

    @staticmethod
    def import_data_with_metadata(
            path: str,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Tuple[str, int]], List[int]]:
        """Import capture data and return preserved segment boundaries."""
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            if not header or header[0] != 'Time':
                raise ValueError("Invalid CSV format — expected 'Time' as first column")

            data_names = header[1:]
            param_names = [name for name in data_names
                           if name != _SEGMENT_COLUMN]
            if not param_names:
                raise ValueError("No parameter columns found")

            rows = list(reader)

        if not rows:
            raise ValueError("CSV file contains no data rows")

        time_arr = np.array([float(row[0]) for row in rows])
        params = {}
        for pname in param_names:
            col_idx = header.index(pname)
            params[pname] = np.array([float(row[col_idx]) for row in rows])

        segment_breaks: List[int] = []
        if _SEGMENT_COLUMN in data_names:
            segment_col = header.index(_SEGMENT_COLUMN)
            segment_ids = np.array(
                [int(float(row[segment_col])) for row in rows], dtype=np.int64)
            segment_breaks = (np.where(np.diff(segment_ids) != 0)[0] + 1).tolist()

        traces = []
        for pname in param_names:
            m = _PARAM_PATTERN.match(pname)
            if m:
                traces.append((m.group(1), int(m.group(2))))
            else:
                traces.append((pname, 0))

        return time_arr, params, traces, segment_breaks
