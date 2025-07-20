from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from typing import Dict, Any

import psutil

from .base_logger import BaseStdoutLogger
from .display_manager import PROCESS_LAYOUT_NAME

class ProcessStdoutLogger(BaseStdoutLogger):
    """
    Stdout logger for Process-level (self PID) CPU and RAM usage metrics.
    Displays live resource usage of your TraceML + training script process.
    """

    def __init__(self):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT_NAME)

        self._latest_data = {
            "process_cpu_percent": 0.0,
            "process_ram_mb": 0.0,
        }

        # Detect system CPU topology at logger initialization
        self.logical_cores = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)
        self.hyperthreaded = self.logical_cores > self.physical_cores
        self.threads_per_core = (
            self.logical_cores // self.physical_cores
            if self.physical_cores else 1
        )

    def _get_panel_renderable(self) -> Panel:
        """
        Generates the Rich Panel for live display:
        Example: "CPU: 22.5% | RAM: 1450MB"
        """
        cpu_val = self._latest_data.get("process_cpu_percent", 0.0)
        ram_val = self._latest_data.get("process_ram_mb", 0.0)

        cpu_display_str = f"CPU ({self.logical_cores} cores): {cpu_val:.1f}%"
        ram_display_str = f"RAM: {ram_val:.0f}MB"

        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(justify="center", style="bold magenta")
        table.add_column(justify="center", style="bold cyan")
        table.add_row(cpu_display_str, ram_display_str)

        return Panel(
            table,
            title="Live Process Metrics",
            title_align="center",
            border_style="dim white",
            width=80,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Logs the final process-level resource summary after the live display.
        """
        print(f"\n[TraceML][{self.name}] Final Summary:")
        for key, value in summary.items():
            if "percent" in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.1f}%")
            elif "_mb" in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.2f} MB")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        print("-" * 40)
