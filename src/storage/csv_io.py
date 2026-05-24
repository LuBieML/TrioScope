import csv
import re
from typing import Dict, List, Tuple
import numpy as np

class CSVStorage:
    @staticmethod
    def export_data(path: str, time_data: np.ndarray, params_data: Dict[str, np.ndarray]) -> None:
        """Export time and parameter data to a CSV file."""
        param_names = list(params_data.keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'] + param_names)
            for i in range(len(time_data)):
                row = [round(time_data[i], 6)] + [params_data[p][i] for p in param_names]
                writer.writerow(row)

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

        # Parse column names like "MPOS(0)" -> param="MPOS", axis=0
        param_pattern = re.compile(r'^(.+)\((\d+)\)$')
        traces = []
        for pname in param_names:
            m = param_pattern.match(pname)
            if m:
                traces.append((m.group(1), int(m.group(2))))
            else:
                traces.append((pname, 0))

        return time_arr, params, traces
