from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.console import Console
from typing import Dict, Any

from .base_logger import BaseStdoutLogger
from .display_manager import SYSTEM_LAYOUT_NAME


class SystemStdoutLogger(BaseStdoutLogger):
    """
    Stdout logger for System (CPU, RAM) usage metrics.
    Displays metrics in the top panel of the live-updating terminal display.
    """

    def __init__(self):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT_NAME)
        # Initialize default values for the live metrics panel
        self._latest_data = {
            "cpu_percent": 0.0,
            "ram_used_mb": 0.0,
            "ram_total_mb": 0.0,
        }

    def _get_panel_renderable(self) -> Panel:
        """
        Generates the Rich Panel for the live metrics display (e.g., "CPU: X% | RAM: Y.YGB").
        """
        cpu_val = self._latest_data.get("cpu_percent", 0.0)
        ram_val = self._latest_data.get("ram_used_mb", 0.0)
        ram_total = self._latest_data.get("ram_total_mb", 0.0)

        ram_display_str = f"{ram_val:.0f}MB/{ram_total:.0f}MB"

        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(justify="center", style="bold green")
        table.add_column(justify="center", style="bold blue")
        table.add_row(f"CPU: {cpu_val:.1f}%", f"RAM: {ram_display_str}")

        return Panel(
            table,
            title="Live System Metrics",
            title_align="center",
            border_style="dim white",
            width=80,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Logs the final summary as a formatted Rich panel using pink styling.
        Should be called after Rich Live display has stopped.
        """
        console = Console()
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold bright_red")
        table.add_column(justify="center", style="bright_red", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        for key, value in summary.items():
            display_key = key.replace('_', ' ').upper()
            if "percent" in key:
                display_value = f"{value:.1f}%"
            elif "ram" in key or "_mb" in key:
                display_value = f"{value:.2f} MB"
            else:
                display_value = str(value)
            table.add_row(display_key, "[bright_red]|[/bright_red]", display_value)

        panel = Panel(
            table, title=f"[bold bright_red]{self.name} - Final Summary", border_style="bright_red",
        )
        console.print(panel)
