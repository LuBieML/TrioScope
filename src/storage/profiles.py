from typing import List
from PySide6.QtCore import QSettings
from models.trace_config import TraceConfig

class ProfileStore:
    def __init__(self, organization: str = "TrioScope", application: str = "ParameterScope", filename: str = None):
        self.organization = organization
        self.application = application
        self.filename = filename

    def _get_settings(self) -> QSettings:
        if self.filename:
            return QSettings(self.filename, QSettings.IniFormat)
        return QSettings(self.organization, self.application)

    def get_profile_names(self) -> List[str]:
        s = self._get_settings()
        names = []
        count = int(s.value("profiles/count", 0))
        for i in range(count):
            name = s.value(f"profiles/{i}/name", None)
            if name:
                names.append(str(name))
        return names

    def save_profile(self, name: str, traces: List[TraceConfig]) -> None:
        s = self._get_settings()
        names = self.get_profile_names()
        if name not in names:
            names.append(name)
        
        s.setValue("profiles/count", len(names))
        for i, n in enumerate(names):
            s.setValue(f"profiles/{i}/name", n)
            
        s.setValue(f"profiles/data/{name}/count", len(traces))
        for i, t in enumerate(traces):
            s.setValue(f"profiles/data/{name}/{i}/param", t.param)
            s.setValue(f"profiles/data/{name}/{i}/axis", t.axis)
            s.setValue(f"profiles/data/{name}/{i}/enabled", "true" if t.enabled else "false")
            s.setValue(f"profiles/data/{name}/{i}/fft", "true" if t.fft else "false")

    def load_profile(self, name: str) -> List[TraceConfig]:
        s = self._get_settings()
        count = int(s.value(f"profiles/data/{name}/count", 0))
        traces = []
        for i in range(count):
            param = str(s.value(f"profiles/data/{name}/{i}/param", "MPOS"))
            axis = int(s.value(f"profiles/data/{name}/{i}/axis", 0))
            enabled = s.value(f"profiles/data/{name}/{i}/enabled", "true") == "true"
            fft = s.value(f"profiles/data/{name}/{i}/fft", "false") == "true"
            traces.append(TraceConfig(param=param, axis=axis, enabled=enabled, fft=fft))
        return traces

    def delete_profile(self, name: str) -> None:
        s = self._get_settings()
        names = self.get_profile_names()
        if name in names:
            names.remove(name)
        
        s.setValue("profiles/count", len(names))
        for i, n in enumerate(names):
            s.setValue(f"profiles/{i}/name", n)
            
        # Remove group
        s.beginGroup(f"profiles/data/{name}")
        s.remove("")
        s.endGroup()

    def rename_profile(self, old_name: str, new_name: str) -> None:
        traces = self.load_profile(old_name)
        self.save_profile(new_name, traces)
        self.delete_profile(old_name)
