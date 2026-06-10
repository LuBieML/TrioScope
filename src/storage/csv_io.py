import csv
import re
from typing import Dict, List, Tuple
import numpy as np

# Column names like "MPOS(0)" -> param="MPOS", axis=0
_PARAM_PATTERN = re.compile(r'^(.+)\((\d+)\)$')


class CSVStorage:
    @staticmethod
    def export_data(path: str, time_data: np.ndarray, params_data: Dict[str, np.ndarray]) -> None:
        """Export time and parameter data to a CSV file."""
        param_names = list(params_data.keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'] + param_names)
            # writerows over zipped columns avoids per-row list building in
            # Python — noticeably faster for 100k+ sample captures.
            columns = [np.round(time_data, 6)]
            columns.extend(params_data[p] for p in param_names)
            writer.writerows(zip(*columns))

    @staticmethod
    def import_data(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Tuple[str, int]]]:
        """
        Import time and parameter data from a CSV file.
        
        Returns:
            Tuple of (time_array, params_dict, list_of_trace_tuples)
            where list_of_trace_tuples is a list of (param_name, axis) parsed from headers.
        """
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            if not header or header[0] != 'Time':
                raise ValueError("Invalid CSV format — expected 'Time' as first column")

            param_names = header[1:]
            if not param_names:
                raise ValueError("No parameter columns found")

            rows = list(reader)

        if not rows:
            raise ValueError("CSV file contains no data rows")

        time_arr = np.array([float(row[0]) for row in rows])
        params = {}
        for col_idx, pname in enumerate(param_names, start=1):
            params[pname] = np.array([float(row[col_idx]) for row in rows])

        traces = []
        for pname in param_names:
            m = _PARAM_PATTERN.match(pname)
            if m:
                traces.append((m.group(1), int(m.group(2))))
            else:
                traces.append((pname, 0))

        return time_arr, params, traces
