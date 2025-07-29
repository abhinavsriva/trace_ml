from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from typing import Dict, Any
from rich.console import Console

import psutil

from .base_logger import BaseStdoutLogger
from .display_manager import PROCESS_LAYOUT_NAME

class ProcessStdoutLogger(BaseStdoutLogger):
    """
    Stdout logger for Process-level (self PID) CPU, RAM and GPU usage metrics.
    Displays live resource usage of your TraceML + training script process.
    """

    def __init__(self):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT_NAME)

        self._latest_data = {
            "process_cpu_percent": 0.0,
            "process_ram_mb": 0.0,
            "process_gpu_mem_mb": None
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
        gpu_mem = self._latest_data.get("process_gpu_memory_mb")

        cpu_display_str = f"CPU ({self.logical_cores} cores): {cpu_val:.1f}%"
        ram_display_str = f"RAM: {ram_val:.0f}MB"

        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(justify="center", style="bold magenta")
        table.add_column(justify="center", style="bold cyan")
        table.add_row(cpu_display_str, ram_display_str)
        if gpu_mem is not None:
            gpu_display_str = f"GPU Memory: {gpu_mem:.0f}MB"
            table.add_row("", gpu_display_str)

        return Panel(
            table,
            title="Live Process Metrics",
            title_align="center",
            border_style="dim white",
            width=80,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Logs the final summary.
        Should be called after Rich Live display has stopped.
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold cyan")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        for key, value in summary.items():
            display_key = key.replace('_', ' ').upper()
            if "percent" in key:
                display_value = f"{value:.1f}%"
            elif "ram" in key or "gpu" in key:
                display_value = f"{value:.2f} MB"
            else:
                display_value = str(value)
            table.add_row(display_key, "[cyan]|[/cyan]", display_value)

        panel = Panel(table, title=f"[bold cyan]{self.name} - Final Summary", border_style="cyan")
        console.print(panel)
