from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.console import Console
from typing import Dict, Any, List

from .base_logger import BaseStdoutLogger
from .display_manager import MODEL_SUMMARY_LAYOUT_NAME, MODEL_HISTORY_LAYOUT_NAME
from .display_manager import StdoutDisplayManager


class LayerMemoryStdoutLogger(BaseStdoutLogger):
    """
    Logger that visualizes:
    - The current model's per-layer parameter memory.
    - A scrollable table of previously detected models.
    """

    def __init__(self):
        super().__init__(name="Layer Memory", layout_section_name=MODEL_SUMMARY_LAYOUT_NAME)

        # Register history panel separately in its own layout section
        StdoutDisplayManager.register_layout_content(
            MODEL_HISTORY_LAYOUT_NAME, self._get_history_renderable
        )

        self._latest_snapshot: Dict[str, Any] = {}
        self._history_snapshots: List[Dict[str, Any]] = []

    def log(self, snapshot: Dict[str, Any]):
        """
        Update both live snapshot and model history.
        """
        self._latest_snapshot = snapshot

        signature = snapshot.get("model_signature", None)
        if signature and signature not in {s.get("model_signature", None) for s in self._history_snapshots}:
            self._history_snapshots.append(snapshot)

        StdoutDisplayManager.update_display()

    def _get_panel_renderable(self) -> Panel:
        """
        Live snapshot of current model's memory usage.
        """
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Layer", justify="left")
        table.add_column("Memory (MB)", justify="right")

        layer_data = self._latest_snapshot.get("layer_memory_mb", {})
        for layer_name, mem_mb in layer_data.items():
            table.add_row(layer_name, f"{mem_mb:.4f}")

        total_mb = self._latest_snapshot.get("total_memory_mb", 0.0)
        model_index = self._latest_snapshot.get("model_index", "No model found")

        return Panel(
            Group(table),
            title=f"Live Model #{model_index} – Total: {total_mb:.2f} MB",
            border_style="cyan",
            width=80
        )

    def _get_history_renderable(self) -> Panel:
        """
        Scrollable history of previously discovered models,
        each displayed as a separate panel like the live model.
        """
        panels = []

        for snapshot in reversed(self._history_snapshots[-20:]):  # Last 20 models
            index = snapshot.get("model_index", "n/a")
            total_mb = snapshot.get("total_memory_mb", 0.0)
            layer_data = snapshot.get("layer_memory_mb", {})
            signature = snapshot.get("model_signature", "")[:50]

            table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
            table.add_column("Layer", justify="left")
            table.add_column("Memory (MB)", justify="right")

            for layer_name, mem_mb in layer_data.items():
                table.add_row(layer_name, f"{mem_mb:.4f}")

            panel = Panel(
                table,
                title=f"Model #{index} – Total: {total_mb:.2f} MB",
                subtitle=f"[dim]Signature: {signature}",
                border_style="cyan",
                width=80
            )
            panels.append(panel)

        return Panel(
            Group(*panels, "\n"),
            title="Model Snapshots (Recent 20)",
            border_style="dim white",
            width=120
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Logs the final summary using Rich.
        Should be called after Rich Live display has stopped.
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold magenta3")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        keys_to_display = [
            "total_models_seen",
            "total_samples_taken",
            "average_model_memory_mb",
            "peak_model_memory_mb"
        ]

        for key in keys_to_display:
            value = summary.get(key, 0)
            display_key = key.replace('_', ' ').upper()
            if "memory" in key:
                display_value = f"{value:.2f} MB"
            else:
                display_value = str(value)
            table.add_row(display_key, "[magenta3]|[/magenta3]", display_value)

        panel = Panel(table, title=f"[bold magenta3]{self.name} - Final Summary", border_style="magenta3")
        console.print(panel)
