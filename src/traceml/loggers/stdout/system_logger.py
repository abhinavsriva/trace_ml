from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from typing import Dict, Any

from .base_logger import BaseStdoutLogger
from .display_manager import LIVE_METRICS_PANEL_NAME

class SystemStdoutLogger(BaseStdoutLogger):
    """
    Stdout logger for System (CPU, RAM) usage metrics.
    Displays metrics in the top panel of the live-updating terminal display.
    """

    def __init__(self):
        super().__init__(name="System", panel_name=LIVE_METRICS_PANEL_NAME)
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

        # Example of a single-row Table for better alignment:
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(justify="center", style="bold green")
        table.add_column(justify="center", style="bold blue")
        table.add_column(justify="center", style="bold magenta")
        table.add_row(
            f"CPU: {cpu_val:.1f}%",
            f"RAM: {ram_display_str}"
        )

        return Panel(
            table,
            title="Live Metrics",
            title_align="center",
            border_style="dim white",
            width=80
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Logs the final System (CPU/RAM) summary to the console after the live display stops.
        """
        # The live display is managed by the manager, so this is for post-mortem logging
        print("\n[TraceML][System] Final Summary:")
        for key, value in summary.items():
            if "percent" in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.1f}%")
            elif "_mb" in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.2f} MB")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        print("-" * 40)