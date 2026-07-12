import json

from PySide6.QtCore import QSettings
from models.app_settings import AppSettings, ConnectionSettings, CaptureSettings, PlotSettings, DisplaySettings
from models.axis_parameter_config import AxisParameterConfig
from models.trace_config import TraceConfig

class SettingsStore:
    def __init__(self, organization: str = "TrioScope", application: str = "ParameterScope", filename: str = None):
        self.organization = organization
        self.application = application
        self.filename = filename

    def _get_settings(self) -> QSettings:
        if self.filename:
            return QSettings(self.filename, QSettings.IniFormat)
        return QSettings(self.organization, self.application)

    def load(self) -> AppSettings:
        s = self._get_settings()
        app_settings = AppSettings()

        # Connection
        app_settings.connection.ip = str(s.value("connection/ip", "192.168.0.245"))

        # Configuration
        app_settings.capture.sample_period = str(s.value("config/sample_period", "1"))
        app_settings.capture.duration = str(s.value("config/duration", "5.0"))
        app_settings.capture.table_start = str(s.value("config/table_start", "0"))
        app_settings.capture.use_end_of_table = s.value("config/use_end_of_table", "true") == "true"
        app_settings.capture.capture_mode = str(s.value("config/capture_mode", "continuous"))
        app_settings.capture.external_trigger = s.value("config/external_trigger", "false") == "true"
        if app_settings.capture.capture_mode == "external":
            app_settings.capture.capture_mode = "single"
            app_settings.capture.external_trigger = True

        # Display / plot settings
        app_settings.display.plot_mode = str(s.value("display/plot_mode", "time"))
        # Migration: old 'fft' global mode -> 'time' with per-trace FFT
        migrate_global_fft = (app_settings.display.plot_mode == 'fft')
        if migrate_global_fft:
            app_settings.display.plot_mode = 'time'
        
        app_settings.display.window_duration = float(s.value("display/window_duration", 5.0))
        app_settings.display.lock_x_axis = s.value("display/lock_x_axis", "true") == "true"
        legacy_path_scale = s.value("display/path_grid_scale", 1.0)
        app_settings.display.path_view_scale = float(
            s.value("display/path_view_scale", legacy_path_scale)
        )
        app_settings.plot.line_width = float(s.value("plot/line_width", 1.0))
        app_settings.plot.grid_alpha = float(s.value("plot/grid_alpha", 0.3))
        app_settings.plot.bg_color = str(s.value("plot/bg_color", "#0A0A0A"))

        # Traces
        num_traces = int(s.value("traces/count", 0))
        for i in range(num_traces):
            param = str(s.value(f"traces/{i}/param", "MPOS")).strip() or "MPOS"
            axis = int(s.value(f"traces/{i}/axis", 0))
            enabled = s.value(f"traces/{i}/enabled", "true") == "true"
            fft = s.value(f"traces/{i}/fft", "false") == "true" or migrate_global_fft
            
            app_settings.traces.append(TraceConfig(
                param=param,
                axis=axis,
                enabled=enabled,
                fft=fft
            ))

        # Restore saved per-axis drive profiles
        num_profiles = int(s.value("ai/drive_profiles/count", 0))
        for i in range(num_profiles):
            axis_val = s.value(f"ai/drive_profiles/{i}/axis", None)
            if axis_val is None:
                continue
            axis = int(axis_val)
            
            def _int_or_none(value) -> int | None:
                if value is None or value == "":
                    return None
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None

            app_settings.drive_profiles[axis] = {
                "drive_type": s.value(f"ai/drive_profiles/{i}/drive_type", "None"),
                "pn100": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn100")),
                "pn101": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn101")),
                "pn102": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn102")),
                "pn103": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn103")),
                "pn104": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn104")),
                "pn105": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn105")),
                "pn106": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn106")),
                "pn112": _int_or_none(s.value(f"ai/drive_profiles/{i}/pn112")),
            }

        # Axis setup grid. A JSON payload keeps the settings schema aligned
        # with the tab's standalone .json import/export format.
        if s.contains("axis_parameters/json"):
            try:
                raw_axis_parameters = json.loads(str(s.value("axis_parameters/json", "[]")))
                if not isinstance(raw_axis_parameters, list):
                    raise ValueError("axis_parameters/json must contain a list")
                app_settings.axis_parameters = [
                    AxisParameterConfig.from_dict(item) for item in raw_axis_parameters
                ]
                axes = [config.axis for config in app_settings.axis_parameters]
                if len(axes) != len(set(axes)):
                    raise ValueError("axis_parameters/json contains duplicate axes")
            except (json.JSONDecodeError, TypeError, ValueError):
                app_settings.axis_parameters = []

        return app_settings

    def save(self, settings: AppSettings) -> None:
        s = self._get_settings()

        # Connection
        s.setValue("connection/ip", settings.connection.ip)

        # Configuration
        s.setValue("config/sample_period", settings.capture.sample_period)
        s.setValue("config/duration", settings.capture.duration)
        s.setValue("config/table_start", settings.capture.table_start)
        s.setValue("config/use_end_of_table", "true" if settings.capture.use_end_of_table else "false")
        s.setValue("config/capture_mode", settings.capture.capture_mode)
        s.setValue("config/external_trigger", "true" if settings.capture.external_trigger else "false")

        # Display / plot settings
        s.setValue("display/plot_mode", settings.display.plot_mode)
        s.setValue("display/window_duration", settings.display.window_duration)
        s.setValue("display/lock_x_axis", "true" if settings.display.lock_x_axis else "false")
        s.setValue("display/path_view_scale", settings.display.path_view_scale)
        s.setValue("plot/line_width", settings.plot.line_width)
        s.setValue("plot/grid_alpha", settings.plot.grid_alpha)
        s.setValue("plot/bg_color", settings.plot.bg_color)

        # Traces
        s.setValue("traces/count", len(settings.traces))
        for i, t in enumerate(settings.traces):
            s.setValue(f"traces/{i}/param", str(t.param or "MPOS").strip() or "MPOS")
            s.setValue(f"traces/{i}/axis", t.axis)
            s.setValue(f"traces/{i}/enabled", "true" if t.enabled else "false")
            s.setValue(f"traces/{i}/fft", "true" if t.fft else "false")

        # Drive profiles
        s.setValue("ai/drive_profiles/count", len(settings.drive_profiles))
        for i, (axis, profile_dict) in enumerate(settings.drive_profiles.items()):
            s.setValue(f"ai/drive_profiles/{i}/axis", axis)
            for key, val in profile_dict.items():
                s.setValue(f"ai/drive_profiles/{i}/{key}", "" if val is None else val)

        s.setValue(
            "axis_parameters/json",
            json.dumps([config.to_dict() for config in settings.axis_parameters]),
        )
