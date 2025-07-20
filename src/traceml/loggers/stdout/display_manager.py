from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from typing import Dict, Any, Callable, Optional, List
import threading
import sys

# Define constants for panel names to ensure consistency
LIVE_METRICS_PANEL_NAME = "live_metrics"
CURRENT_MODEL_SUMMARY_PANEL_NAME = "current_model_summary"
MODEL_SNAPSHOTS_TABLE_PANEL_NAME = "model_snapshots_table"


class StdoutDisplayManager:
    """
    Manages a single shared Rich Live display and a dynamic Layout for all stdout loggers.
    """
    _console: Console = Console()
    _live_display: Optional[Live] = None
    _layout: Layout = Layout(name="root")

    # Registry for functions that generate content for specific layout sections
    # Key: layout_panel_name (e.g., "live_metrics")
    # Value: Callable[[], Renderable] - a function that returns the latest renderable for that panel
    _panel_content_fns: Dict[str, Callable[[], Any]] = {}

    # For thread safety if multiple threads update _panel_content_fns
    _lock = threading.Lock()

    @classmethod
    def _create_initial_layout(cls):
        """
        Defines the improved structure of the Rich Layout with flexible ratios.
        """
        cls._layout.split_column(
            Layout(name=LIVE_METRICS_PANEL_NAME, ratio=1),
            Layout(name=CURRENT_MODEL_SUMMARY_PANEL_NAME, ratio=2),
            Layout(name=MODEL_SNAPSHOTS_TABLE_PANEL_NAME, ratio=3)
        )

        cls._layout[LIVE_METRICS_PANEL_NAME].update(
            Panel(Text("Initializing Live Metrics...", justify="center"))
        )
        cls._layout[CURRENT_MODEL_SUMMARY_PANEL_NAME].update(
            Panel(Text("Waiting for Model Data...", justify="center"))
        )
        cls._layout[MODEL_SNAPSHOTS_TABLE_PANEL_NAME].update(
            Panel(Text("No Layer Snapshots Yet...", justify="center"))
        )

    @classmethod
    def start_display(cls):
        """Starts the shared Rich Live display if not already running."""
        with cls._lock:
            if cls._live_display is None:
                cls._create_initial_layout()
                cls._live_display = Live(
                    cls._layout,
                    console=cls._console,
                    auto_refresh=False, # We'll manage refresh manually
                    transient=False,    # Keep output after stop
                    screen=True         # Use full screen if possible for better experience
                )
                try:
                    cls._live_display.start()
                    cls._live_display.refresh()
                except Exception as e:
                    print(f"[TraceML] Failed to start shared live display: {e}", file=sys.stderr)
                    cls._live_display = None # Reset if failed to start

    @classmethod
    def stop_display(cls):
        """Stops the shared Rich Live display."""
        with cls._lock:
            if cls._live_display:
                try:
                    cls._live_display.stop()
                except Exception as e:
                    print(f"[TraceML] Error stopping live display: {e}", file=sys.stderr)
                finally:
                    cls._live_display = None
                    cls._panel_content_fns.clear()
                    # Re-initialize layout to reset state for next run
                    cls._layout = Layout(name="root")


    @classmethod
    def register_panel_content(cls, panel_name: str, content_fn: Callable[[], Any]):
        """
        Registers a function that provides content for a specific layout panel.
        """
        with cls._lock:
            if cls._layout.get(panel_name) is None:
                print(f"[TraceML] WARNING: Layout panel '{panel_name}' not found. Cannot register content.", file=sys.stderr)
                return
            cls._panel_content_fns[panel_name] = content_fn


    @classmethod
    def update_display(cls):
        """
        Triggers an update of the entire live display by calling all registered
        content functions and updating the layout.
        """
        with cls._lock:
            if cls._live_display is None:
                # If display fails to start log to console directly
                # print("Live display not active. Logging directly to console.", file=sys.stderr)
                # Fallback to direct print for each registered panel (simplified)
                # This fallback needs a better design to show *all* current state
                return # For now, just exit if live display isn't running

            try:
                for panel_name, content_fn in cls._panel_content_fns.items():
                    try:
                        renderable = content_fn()
                        if renderable is not None:
                             cls._layout[panel_name].update(renderable)
                    except Exception as e:
                        error_panel = Panel(
                            f"[red]Error rendering {panel_name}: {e}[/red]",
                            title=f"[bold red]Render Error: {panel_name}[/bold red]",
                            border_style="red"
                        )
                        cls._layout[panel_name].update(error_panel)
                        print(f"[TraceML] Error in rendering content for panel {panel_name}: {e}", file=sys.stderr)

                cls._live_display.refresh() # Only refresh once per update cycle
            except Exception as e:
                print(f"[TraceML] Error updating live display: {e}", file=sys.stderr)