"""Aerospace Ground Station - Textual TUI Dashboard with Visual Graph."""

from __future__ import annotations

import asyncio
import aiosqlite
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Interrupt
from llama_cpp import Llama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, Grid
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Log,
    RichLog,
    Static,
    ProgressBar,
    Tree as TreeWidget,
)

from logistics.graph import build_logistics_graph
from logistics.state import LogisticsState
from main import (
    MissionState,
    approval_gate,
    approval_router,
    build_async_checkpointer,
    build_checkpointer,
    build_graph,
    end_node,
    extract_tool_calls,
    format_tool_outputs,
    get_certifications,
    load_model,
    logistics_adapter_in,
    logistics_adapter_out,
    systems_adapter_in,
    systems_adapter_out,
    tool_router,
    tools_node,
    tools_return_router,
    triage_node,
    triage_router,
)
from systems.graph import build_systems_graph
from systems.state import SystemsState
from tools import check_fuel, log_anomaly, read_pressure


# ============================================================================
# Quick-Start Examples (Step 2: Integrated Examples)
# ============================================================================

MISSION_EXAMPLES = [
    "Check fuel levels for the RCS modules.",
    "Read pressure for reactor sensor RCS-4.",
    "Log a critical reactor failure anomaly.",
]

CERTIFICATION_EXAMPLES = [
    "Certify the 'Starship-X' project: a new experimental engine with novel solar propulsion.",
    "Process PRC for 'Satellite-A': a routine resupply orbital satellite (minor project).",
    "Validate 'Mars-Home' safety compliance: high-complexity habitat for 10 astronauts.",
]

ALL_EXAMPLES = MISSION_EXAMPLES + CERTIFICATION_EXAMPLES


class QuickStartWidget(Static):
    """Widget showing quick-start examples that can cycle dynamically."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._example_index = 0

    def on_mount(self) -> None:
        """Start cycling through examples."""
        self.set_interval(8.0, self.cycle_example)
        self.update_example()

    def update_example(self) -> None:
        """Update to show current example."""
        example = ALL_EXAMPLES[self._example_index]
        category = "🚀 Mission" if example in MISSION_EXAMPLES else "📋 Certification"
        self.update(
            f"💡 [bold cyan]Quick Start:[/bold cyan] {category}\n"
            f"   [dim]Type:[/dim] {example}"
        )

    def cycle_example(self) -> None:
        """Cycle to next example."""
        self._example_index = (self._example_index + 1) % len(ALL_EXAMPLES)
        self.update_example()


# ============================================================================
# Graph Node Definitions for Visualization
# ============================================================================

class NodeStatus(Enum):
    """Node execution status."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class GraphNode:
    """Represents a node in the mission control graph."""
    name: str
    display_name: str
    emoji: str
    color: str
    description: str
    node_type: str  # router, adapter, specialist, executor, hitl, terminal
    status: NodeStatus = NodeStatus.PENDING


# Main Graph Nodes
MAIN_GRAPH_NODES = [
    GraphNode("triage_node", "Triage", "🔍", "cyan", "Analyzes query & routes to specialist", "router"),
    GraphNode("logistics_adapter_in", "Logistics In", "📥", "green", "Maps state to Logistics subgraph", "adapter"),
    GraphNode("logistics", "Logistics Specialist", "🚀", "green", "Fuel, Cargo, Resupply, Docking", "specialist"),
    GraphNode("logistics_adapter_out", "Logistics Out", "📤", "green", "Maps state from Logistics subgraph", "adapter"),
    GraphNode("systems_adapter_in", "Systems In", "📥", "yellow", "Maps state to Systems subgraph", "adapter"),
    GraphNode("systems", "Systems Specialist", "⚙️", "yellow", "Propulsion, Navigation, Life Support", "specialist"),
    GraphNode("systems_adapter_out", "Systems Out", "📤", "yellow", "Maps state from Systems subgraph", "adapter"),
    GraphNode("parallel_handler", "Parallel Handler", "🔀", "magenta", "Routes to both specialists", "router"),
    GraphNode("parallel_merger", "Parallel Merger", "🔗", "magenta", "Merges parallel results", "merger"),
    GraphNode("tools", "Telemetry Tools", "📊", "blue", "Read sensors, check fuel, log anomaly", "executor"),
    GraphNode("approval_gate", "Approval Gate", "🛡️", "red", "Human-in-the-loop approval", "hitl"),
    GraphNode("end_node", "End Node", "🏁", "bright_cyan", "Generates final report", "terminal"),
]

# Project Classification Nodes (Certification Path)
CERTIFICATION_NODES = [
    GraphNode("cert_triage", "Cert Triage", "📋", "bright_magenta", "Classify project type & requirements", "router"),
    GraphNode("cert_aerospace", "Aerospace Cert", "🛸", "bright_blue", "Aerospace compliance checks", "certifier"),
    GraphNode("cert_mission", "Mission Cert", "🎯", "bright_green", "Mission requirements validation", "certifier"),
    GraphNode("cert_safety", "Safety Cert", "⚠️", "bright_red", "Safety & risk assessment", "certifier"),
    GraphNode("cert_final", "Final Cert", "✅", "bright_cyan", "Certification report & approval", "terminal"),
]


# ============================================================================
# Visual Graph Widget
# ============================================================================

class VisualGraphWidget(Static):
    """Visual representation of the graph nodes and connections."""

    def __init__(self, nodes: list[GraphNode], title: str = "Graph Flow", **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.nodes = nodes
        self.title = title
        self._active_node: str | None = None
        self._animation_frame = 0

    def update_node_status(self, node_name: str, status: NodeStatus) -> None:
        """Update the status of a specific node."""
        for node in self.nodes:
            if node.name == node_name:
                node.status = status
                break
        self.refresh()

    def set_active_node(self, node_name: str | None) -> None:
        """Set the currently active node."""
        self._active_node = node_name
        self.refresh()

    def reset_all(self) -> None:
        """Reset all nodes to pending status."""
        for node in self.nodes:
            node.status = NodeStatus.PENDING
        self._active_node = None
        self.refresh()

    def render(self) -> str:
        """Render the visual graph."""
        lines = []
        lines.append(f"┌{'─' * 70}┐")
        lines.append(f"│ {self.title:^68} │")
        lines.append(f"├{'─' * 70}┤")

        for i, node in enumerate(self.nodes):
            # Status indicator with animation
            # Task 4: Human review nodes pulse in ORANGE
            if node.status == NodeStatus.ACTIVE:
                # Animated indicator for active nodes
                anim_chars = ["●", "○", "●", "○"]
                # Use orange for human_review node (Task 4)
                if node.name == "human_review":
                    indicator = f"[orange]{anim_chars[self._animation_frame % len(anim_chars)]}[/orange]"
                else:
                    indicator = f"[{node.color}]{anim_chars[self._animation_frame % len(anim_chars)]}[/{node.color}]"
            elif node.status == NodeStatus.COMPLETED:
                indicator = "[green]✓[/green]"
            elif node.status == NodeStatus.ERROR:
                indicator = "[red]✗[/red]"
            elif node.status == NodeStatus.SKIPPED:
                indicator = "[dim]○[/dim]"
            else:
                indicator = "[dim]○[/dim]"

            # Node name with highlighting if active
            # Task 4: Human review uses orange color
            display_color = "orange" if node.name == "human_review" else node.color
            if node.name == self._active_node or node.status == NodeStatus.ACTIVE:
                name_display = f"[{display_color} bold]{node.display_name}[/{display_color} bold]"
            else:
                name_display = f"[{display_color}]{node.display_name}[/{display_color}]"

            # Type badge
            type_badge = f"[dim]({node.node_type})[/dim]"

            # Description
            desc = f"[dim]{node.description}[/dim]"

            line = f"│ {indicator} {node.emoji} {name_display:<25} {type_badge:<18} {desc:<20} │"
            lines.append(line)

            # Add connection arrow if not last node
            if i < len(self.nodes) - 1:
                next_node = self.nodes[i + 1]
                if node.status == NodeStatus.COMPLETED and next_node.status == NodeStatus.PENDING:
                    arrow = "[dim]│[/dim]"
                elif node.status == NodeStatus.COMPLETED:
                    arrow = "[green]▼[/green]"
                elif node.status == NodeStatus.ACTIVE:
                    arrow = "[cyan]▼[/cyan]"
                else:
                    arrow = "[dim]│[/dim]"
                lines.append(f"│ {arrow:^68} │")

        lines.append(f"└{'─' * 70}┘")

        # Add legend
        lines.append("")
        lines.append("  [cyan]●[/cyan] Active  [green]✓[/green] Complete  [red]✗[/red] Error  [dim]○[/dim] Pending  [orange]●[/orange] Waiting Review")

        return "\n".join(lines)

    def animate(self) -> None:
        """Advance animation frame."""
        self._animation_frame += 1
        self.refresh()


# ============================================================================
# Visual Styles (CSS)
# ============================================================================

DASHBOARD_CSS = """
/* Main Grid Layout */
#app-grid {
    layout: grid;
    grid-size: 3 4;
    grid-gutter: 1;
    grid-rows: 1fr 1fr 1fr 3;
}

/* Graph Visualization Panel */
#graph-visual {
    row-span: 2;
    column-span: 2;
    border: thick $primary;
    border-title-color: $primary;
    border-title-style: bold;
    color: $text;
    background: $surface;
    overflow-y: scroll;
}

/* Log Feed Panel */
#log-feed {
    row-span: 2;
    border: thick $accent;
    border-title-color: $accent;
    border-title-style: bold;
    color: $text;
    background: $surface;
    overflow-y: scroll;
}

/* Specialist View */
#specialist-view {
    border: thick $success;
    border-title-color: $success;
    border-title-style: bold;
    color: $text;
    background: $surface;
}

/* Telemetry Panel */
#telemetry-panel {
    border: thick $warning;
    border-title-color: $warning;
    border-title-style: bold;
    color: $text;
    background: $surface;
}

/* Command Input */
#command-input {
    row-span: 1;
    column-span: 3;
    border: thick $accent;
    border-title-color: $accent;
    border-title-style: bold;
}

/* Status Bar */
#status-bar {
    column-span: 3;
    color: $success;
    text-style: bold;
}

/* Path Selection Panel */
#path-selection {
    border: thick $primary;
    border-title-color: $primary;
    border-title-style: bold;
    background: $surface;
    padding: 1;
}

/* Path Button Styles */
.path-button {
    width: 100%;
    margin: 1 0;
}

.path-button-primary {
    background: $primary;
    color: $text;
}

.path-button-secondary {
    background: $warning;
    color: $text;
}

/* Welcome Banner */
#welcome-banner {
    border: thick $success;
    border-title-color: $success;
    border-title-style: bold;
    background: $surface;
    text-align: center;
    padding: 1;
}

/* Node Status Indicators */
.node-active {
    color: $primary;
    text-style: bold;
    text-style: blink;
}

.node-completed {
    color: $success;
}

.node-pending {
    color: $text-muted;
}

.node-error {
    color: $error;
}

/* Animation Classes */
.pulse {
    text-style: bold;
}

.glow {
    color: $primary;
    text-style: bold;
}

/* Log Entry Styles */
.log-entry {
    color: $text;
}

.log-triage {
    color: $primary;
}

.log-logistics {
    color: $success;
}

.log-systems {
    color: $warning;
}

.log-tools {
    color: $accent;
}

.log-approval {
    color: $error;
}

.log-end {
    color: $primary;
}

.log-cert {
    color: $primary;
}

.specialist-active {
    color: $success;
    text-style: bold;
}

.telemetry-value {
    color: $accent;
}

.telemetry-label {
    color: $text-muted;
}

.command-prompt {
    color: $success;
}

/* Approval Modal */
.approval-modal {
    align: center middle;
    background: $surface;
    border: thick $error;
    padding: 2 4;
}

.approval-title {
    color: $error;
    text-style: bold;
    text-align: center;
}

.approval-text {
    color: $text;
    text-align: center;
    margin: 1 0;
}

.approval-buttons {
    align: center middle;
    margin-top: 1;
}

.approval-buttons Button {
    margin: 0 1;
}

/* Session Info */
.session-info {
    color: $text-muted;
    text-align: right;
}

/* Progress Animation */
.progress-bar {
    color: $primary;
}

/* Path Info Cards */
.path-card {
    border: solid $primary;
    padding: 1;
    margin: 1;
}

.path-card-secondary {
    border: solid $warning;
    padding: 1;
    margin: 1;
}

/* Certification Approval Modal (Task 3) */
.cert-approval-modal {
    align: center middle;
    background: $surface;
    border: thick $warning;
    padding: 2 4;
    width: 80;
    height: auto;
}

.cert-approval-title {
    color: $warning;
    text-style: bold;
    text-align: center;
}

.cert-approval-spacer {
    height: 1;
}

.cert-approval-label {
    color: $text-muted;
    text-style: bold;
}

.cert-approval-text {
    color: $text;
    text-align: left;
    margin: 0 0;
}

.cert-approval-reasoning {
    color: $accent;
    text-align: left;
}

.cert-approval-warning {
    color: $error;
    text-style: italic;
    text-align: center;
}

.cert-approval-question {
    color: $text;
    text-style: bold;
    text-align: center;
}

.cert-approval-buttons {
    align: center middle;
    margin-top: 1;
}

.cert-approval-buttons Button {
    margin: 0 2;
}
"""


# ============================================================================
# Custom Widgets
# ============================================================================

class AnimatedBanner(Static):
    """Animated welcome banner with pulsing effect."""

    def __init__(self, text: str, **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self._frame = 0
        self._base_text = text

    def on_mount(self) -> None:
        """Start animation on mount."""
        self.set_interval(0.5, self.animate)

    def animate(self) -> None:
        """Animate the banner."""
        self._frame += 1
        frames = ["█", "▓", "▒", "░", "▒", "▓"]
        char = frames[self._frame % len(frames)]
        self.update(f"{char} {self._base_text} {char}")


class LogFeed(RichLog):
    """Scrolling log feed showing graph node execution."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(markup=True, **kwargs)
        self.auto_scroll = True

    def add_entry(self, node: str, message: str, timestamp: bool = True) -> None:
        """Add a log entry with node-specific styling."""
        ts = datetime.now().strftime("%H:%M:%S") if timestamp else ""
        prefix = f"[{ts}] " if timestamp else ""

        # Map node names to colors
        color_map = {
            "triage": "cyan",
            "logistics": "green",
            "systems": "yellow",
            "tools": "blue",
            "approval": "red",
            "end": "bright_cyan",
            "cert": "bright_magenta",
            "parallel": "magenta",
        }

        node_lower = node.lower()
        color = "primary"
        for key, val in color_map.items():
            if key in node_lower:
                color = val
                break

        styled_node = f"[bold {color}]{node}[/]"
        entry = f"{prefix}{styled_node}: {message}"
        self.write(entry)

    def add_raw(self, text: str) -> None:
        """Add raw text to the log."""
        self.write(text)

    def add_divider(self, char: str = "─") -> None:
        """Add a visual divider."""
        self.write(f"[dim]{char * 60}[/dim]")


class SpecialistView(Static):
    """Shows which specialist subgraph is currently active."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("**AWAITING QUERY**", **kwargs)
        self._active = "idle"

    def set_active(self, specialist: str) -> None:
        """Set the active specialist."""
        self._active = specialist
        if specialist == "logistics":
            self.update("🚀 [bold green]LOGISTICS SPECIALIST[/bold green]\n\n📦 Fuel & Cargo\n🛸 Resupply & Docking")
        elif specialist == "systems":
            self.update("⚙️ [bold yellow]SYSTEMS SPECIALIST[/bold yellow]\n\n🔧 Propulsion & Navigation\n💨 Life Support & Reactor")
        elif specialist == "parallel":
            self.update("🔀 [bold magenta]PARALLEL EXECUTION[/bold magenta]\n\n⚡ Both Specialists Active\n📊 Merged Report Pending")
        elif specialist == "certification":
            self.update("📋 [bold bright_magenta]CERTIFICATION MODE[/bold bright_magenta]\n\n🛸 Aerospace Compliance\n🎯 Mission Validation\n⚠️ Safety Assessment")
        elif specialist == "idle":
            self.update("**AWAITING QUERY**")
        else:
            self.update(f"**{specialist.upper()}**")

    @property
    def active(self) -> str:
        return self._active


class TelemetryPanel(Static):
    """Display for fuel levels and sensor data."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._fuel_main = 67
        self._fuel_rcs = 88
        self._pressure = 340
        self._status = "STABLE"
        self._animation_frame = 0

    def on_mount(self) -> None:
        """Start telemetry animation."""
        self.set_interval(1.0, self.animate)

    def animate(self) -> None:
        """Animate telemetry values slightly."""
        self._animation_frame += 1
        # Slight variations for realism
        import random
        self._fuel_main = max(50, min(100, self._fuel_main + random.randint(-1, 1)))
        self._fuel_rcs = max(70, min(100, self._fuel_rcs + random.randint(-1, 1)))
        self.update_display()

    def update_telemetry(self, **kwargs: Any) -> None:
        """Update telemetry values."""
        if "fuel_main" in kwargs:
            self._fuel_main = kwargs["fuel_main"]
        if "fuel_rcs" in kwargs:
            self._fuel_rcs = kwargs["fuel_rcs"]
        if "pressure" in kwargs:
            self._pressure = kwargs["pressure"]
        if "status" in kwargs:
            self._status = kwargs["status"]
        self.update_display()

    def update_display(self) -> None:
        """Update the telemetry display."""
        # Status color based on values
        fuel_color = "green" if self._fuel_main > 50 else "yellow" if self._fuel_main > 25 else "red"
        pressure_color = "green" if self._status == "STABLE" else "yellow" if self._status == "WARNING" else "red"

        self.update(
            f"⛽ **FUEL LEVELS**\n"
            f"   Main Tank: [{fuel_color} bold]{self._fuel_main}%[/{fuel_color} bold]\n"
            f"   RCS Tank:  [{fuel_color} bold]{self._fuel_rcs}%[/{fuel_color} bold]\n\n"
            f"📊 **SENSORS**\n"
            f"   Pressure:  [{pressure_color} bold]{self._pressure} PSI[/{pressure_color} bold]\n"
            f"   Status:    [{pressure_color} bold]{self._status}[/{pressure_color} bold]\n\n"
            f"🛰️ **SYSTEMS**\n"
            f"   Reactor:   [bold green]NOMINAL[/bold green]\n"
            f"   Life Support: [bold green]OK[/bold green]"
        )


class PathSelectionWidget(Static):
    """Widget for selecting between two paths: Mission Graph or Certification."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._selected_path: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("🎯 [bold]SELECT OPERATION MODE[/bold] 🎯", id="path-title")
            yield Label("")
            yield Label("┌─────────────────────────────────────────────────────────────────┐")
            yield Label("│  [1] 🚀 MISSION CONTROL GRAPH                                  │")
            yield Label("│      ├─ Triage & Route Queries                                 │")
            yield Label("│      ├─ Logistics Specialist (Fuel, Cargo, Docking)            │")
            yield Label("│      ├─ Systems Specialist (Propulsion, Navigation)            │")
            yield Label("│      └─ Telemetry Tools & HITL Approval                        │")
            yield Label("├─────────────────────────────────────────────────────────────────┤")
            yield Label("│  [2] 📋 PROJECT CLASSIFICATION & CERTIFICATION                │")
            yield Label("│      ├─ Classify Project Type                                  │")
            yield Label("│      ├─ Aerospace Compliance Checks                            │")
            yield Label("│      ├─ Mission Requirements Validation                        │")
            yield Label("│      └─ Safety & Risk Assessment                               │")
            yield Label("└─────────────────────────────────────────────────────────────────┘")
            yield Label("")
            yield Label("  Press [1] or [2] to select path, or type query to auto-route")

    def set_selected(self, path: str) -> None:
        """Set the selected path."""
        self._selected_path = path
        if path == "mission":
            self.styles.border = ("thick", "green")
        elif path == "certification":
            self.styles.border = ("thick", "yellow")
        self.refresh()

    @property
    def selected_path(self) -> str | None:
        return self._selected_path


class CommandInput(Input):
    """Persistent command input field."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(placeholder="Enter mission control query...", **kwargs)


class ApprovalModal(Screen[bool]):
    """Modal for HITL approval/rejection (mission control)."""

    def __init__(self, pending_action: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pending_action = pending_action

    def compose(self) -> ComposeResult:
        with Container(classes="approval-modal"):
            yield Label("⚠️  APPROVAL REQUIRED  ⚠️", classes="approval-title")
            yield Label(f"\n{self.pending_action}\n", classes="approval-text")
            yield Label("Do you approve this action?", classes="approval-text")
            with Horizontal(classes="approval-buttons"):
                yield Button("[A]pprove", id="approve", variant="success")
                yield Button("[R]eject", id="reject", variant="error")

    @on(Button.Pressed, "#approve")
    def approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def reject(self) -> None:
        self.dismiss(False)

    def on_key(self, event: Any) -> None:
        if event.key.lower() == "a":
            self.approve()
        elif event.key.lower() == "r":
            self.reject()


class CertificationApprovalModal(Screen[bool]):
    """Modal for HITL certification approval/rejection (Task 3).

    Extends ModalScreen to appear when graph hits human_review breakpoint.
    Displays Project Description, PRC Classification results, and [CERTIFY]/[REJECT] buttons.
    """

    def __init__(self, project_description: str, classification_reasoning: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project_description = project_description
        self.classification_reasoning = classification_reasoning

    def compose(self) -> ComposeResult:
        with Container(classes="cert-approval-modal"):
            yield Label("📋  CERTIFICATION REVIEW REQUIRED  📋", classes="cert-approval-title")
            yield Label("", classes="cert-approval-spacer")
            yield Label("Project Description:", classes="cert-approval-label")
            yield Label(self.project_description[:200] + "..." if len(self.project_description) > 200 else self.project_description, classes="cert-approval-text")
            yield Label("", classes="cert-approval-spacer")
            yield Label("PRC Classification: [bold orange]MAJOR[/bold orange]", classes="cert-approval-label")
            yield Label(f"Reasoning: {self.classification_reasoning}", classes="cert-approval-reasoning")
            yield Label("", classes="cert-approval-spacer")
            yield Label("⚠️  Human-in-the-loop review required for certification", classes="cert-approval-warning")
            yield Label("", classes="cert-approval-spacer")
            yield Label("Do you certify this project?", classes="cert-approval-question")
            with Horizontal(classes="cert-approval-buttons"):
                yield Button("[C]ertify", id="certify", variant="success")
                yield Button("[R]eject", id="reject", variant="error")

    @on(Button.Pressed, "#certify")
    def certify(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def reject(self) -> None:
        self.dismiss(False)

    def on_key(self, event: Any) -> None:
        if event.key.lower() == "c":
            self.certify()
        elif event.key.lower() == "r":
            self.reject()


# ============================================================================
# Main Dashboard App
# ============================================================================

class GroundStationDashboard(App):
    """Aerospace Ground Station TUI Dashboard with Visual Graph."""

    CSS = DASHBOARD_CSS
    TITLE = "🚀 TINYGRAPH - AEROSPACE MISSION CONTROL"
    SUB_TITLE = "Triage & Graph Application"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+s", "save_session", "Save Session"),
        Binding("ctrl+r", "reset_graph", "Reset Graph"),
        Binding("ctrl+l", "list_certifications", "List Certs"),
        Binding("f1", "show_help", "Help"),
        Binding("1", "select_mission_path", "Mission Graph"),
        Binding("2", "select_cert_path", "Certification"),
    ]

    # Current operation mode: "mission" or "certification"
    operation_mode: reactive[str] = reactive("mission")

    def __init__(self, model: Llama, checkpointer: SqliteSaver, session_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.checkpointer = checkpointer
        self.session_id = session_id
        self.config = {"configurable": {"thread_id": session_id}}
        self.graph = build_graph(model, checkpointer)
        self.current_specialist = "idle"
        self.pending_action: str | None = None
        self._animation_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="app-grid"):
            # Visual Graph Panel (left, spans 2 rows, 2 columns)
            yield VisualGraphWidget(MAIN_GRAPH_NODES, title="🚀 Mission Control Graph", id="graph-visual")
            # Log Feed (right, spans 2 rows)
            yield LogFeed(id="log-feed", highlight=False)
            # Specialist View
            yield SpecialistView(id="specialist-view")
            # Telemetry Panel
            yield TelemetryPanel(id="telemetry-panel")
            # Command Input (full width)
            yield CommandInput(id="command-input")
            # Status Bar (full width)
            yield Static(f"Session: {self.session_id[:12]}... | Mode: MISSION | Ready", id="status-bar", classes="session-info")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount with quick-start examples (Steps 2 & 3)."""
        log_feed = self.query_one(LogFeed)

        # Welcome banner
        log_feed.add_entry("SYSTEM", "╔═══════════════════════════════════════════════════════════════╗", timestamp=False)
        log_feed.add_entry("SYSTEM", "║  🚀 TINYGRAPH - Aerospace Mission Control Initialized        ║", timestamp=False)
        log_feed.add_entry("SYSTEM", "║  📊 Triage & Graph Application with Visual Dashboard         ║", timestamp=False)
        log_feed.add_entry("SYSTEM", "╚═══════════════════════════════════════════════════════════════╝", timestamp=False)
        log_feed.add_entry("SYSTEM", f"Session ID: {self.session_id}", timestamp=False)
        log_feed.add_entry("SYSTEM", "Press [1] for Mission | [2] for Certification | Ctrl+L for History", timestamp=False)

        # Step 2 & 3: Show quick-start examples in LogFeed
        log_feed.add_divider("─")
        log_feed.add_entry("EXAMPLES", "💡 QUICK START EXAMPLES (copy-paste to try):", timestamp=False)

        log_feed.add_raw("[bold green]🚀 MISSION FLOW:[/bold green]")
        for i, ex in enumerate(MISSION_EXAMPLES, 1):
            log_feed.add_raw(f"  [{i}] {ex}")

        log_feed.add_raw("")
        log_feed.add_raw("[bold bright_magenta]📋 CERTIFICATION FLOW:[/bold bright_magenta]")
        for i, ex in enumerate(CERTIFICATION_EXAMPLES, 1):
            log_feed.add_raw(f"  [{i}] {ex}")

        log_feed.add_divider("─")
        log_feed.add_entry("SYSTEM", "Ready for mission control queries", timestamp=False)

        # Start graph animation
        self._animation_timer = self.set_interval(0.3, self._animate_graph)

        # Focus the command input
        self.query_one(CommandInput).focus()

    def _animate_graph(self) -> None:
        """Animate the visual graph."""
        try:
            graph_widget = self.query_one(VisualGraphWidget)
            graph_widget.animate()
        except:
            pass

    def update_graph_node(self, node_name: str, status: NodeStatus) -> None:
        """Update a node's status in the visual graph."""
        try:
            graph_widget = self.query_one(VisualGraphWidget)
            graph_widget.update_node_status(node_name, status)
            graph_widget.set_active_node(node_name if status == NodeStatus.ACTIVE else None)
        except:
            pass

    def reset_graph_visual(self) -> None:
        """Reset the visual graph to initial state."""
        try:
            graph_widget = self.query_one(VisualGraphWidget)
            graph_widget.reset_all()
        except:
            pass

    def action_select_mission_path(self) -> None:
        """Select the mission control graph path."""
        self.operation_mode = "mission"
        self.update_status_bar()
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("MODE", "Switched to 🚀 MISSION CONTROL GRAPH")
        # Update graph to show mission nodes
        try:
            graph_widget = self.query_one(VisualGraphWidget)
            graph_widget.nodes = MAIN_GRAPH_NODES
            graph_widget.title = "🚀 Mission Control Graph"
            graph_widget.reset_all()
        except:
            pass

    def action_select_cert_path(self) -> None:
        """Select the certification path."""
        self.operation_mode = "certification"
        self.update_status_bar()
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("MODE", "Switched to 📋 PROJECT CLASSIFICATION & CERTIFICATION")
        # Update graph to show certification nodes
        try:
            graph_widget = self.query_one(VisualGraphWidget)
            graph_widget.nodes = CERTIFICATION_NODES
            graph_widget.title = "📋 Project Classification & Certification"
            graph_widget.reset_all()
        except:
            pass

    def update_status_bar(self) -> None:
        """Update the status bar with current mode."""
        mode_text = "MISSION" if self.operation_mode == "mission" else "CERTIFICATION"
        try:
            status_bar = self.query_one("#status-bar")
            status_bar.update(f"Session: {self.session_id[:12]}... | Mode: {mode_text} | Ready")
        except:
            pass

    def action_reset_graph(self) -> None:
        """Reset the graph visual."""
        self.reset_graph_visual()
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("SYSTEM", "Graph visual reset")

    def action_list_certifications(self) -> None:
        """List past project certifications from SQLite (Task 3: Certification Persistence)."""
        log_feed = self.query_one(LogFeed)

        # Get connection from checkpointer
        try:
            conn = self.checkpointer.conn
            certifications = get_certifications(conn)

            log_feed.add_raw("")
            log_feed.add_raw("╔══════════════════════════════════════════════════════════════════╗")
            log_feed.add_raw("║           PAST PROJECT CERTIFICATIONS                            ║")
            log_feed.add_raw("╠══════════════════════════════════════════════════════════════════╣")

            if not certifications:
                log_feed.add_raw("║  No certifications found in database.                            ║")
            else:
                for cert in certifications[:10]:  # Show last 10
                    status_emoji = "✓" if cert["status"] == "Passed" else "✗" if cert["status"] == "Failed" else "○"
                    desc = (cert["project_description"] or "N/A")[:35]
                    log_feed.add_raw(f"║  {status_emoji} [{cert['classification']:<6}] {cert['status']:<8} {desc:<35} ║")

            log_feed.add_raw("╚══════════════════════════════════════════════════════════════════╝")
            log_feed.add_raw("")
            log_feed.add_entry("SYSTEM", f"Listed {len(certifications)} certification records")

        except Exception as e:
            log_feed.add_entry("ERROR", f"Failed to load certifications: {str(e)}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command submission. Type 'quit' to exit."""
        query = event.value.strip()
        if not query:
            return

        # Clear input
        event.input.value = ""

        # Stop session when user types quit
        if query.lower() in ("quit", "exit", "q"):
            log_feed = self.query_one(LogFeed)
            log_feed.add_entry("SYSTEM", "Session ended by user. Goodbye!")
            self.exit()
            return

        # Log the query
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("USER", f"Query received: {query}")

        # Process the query
        await self.process_query(query)

    async def process_query(self, query: str) -> None:
        """Process a mission control query through the graph."""
        log_feed = self.query_one(LogFeed)
        specialist_view = self.query_one(SpecialistView)
        telemetry_panel = self.query_one(TelemetryPanel)

        # Reset visual graph for new query
        self.reset_graph_visual()

        # Add visual divider
        log_feed.add_divider("═")
        log_feed.add_entry("QUERY", f"Processing: {query[:50]}..." + ("..." if len(query) > 50 else ""))

        # Initialize state
        initial_state: MissionState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "next_node": "",
            "next_step": "",
            "tool_attempts": 0,
            "tool_failed": False,
            "tool_error": "",
            "last_tool_caller": "",
            "operator_approved": None,
            "pending_action": None,
        }

        log_feed.add_entry("TRIAGE", "Analyzing query...")

        # Mark triage as active
        self.update_graph_node("triage_node", NodeStatus.ACTIVE)

        try:
            # Stream the graph execution
            async for event in self.graph.astream(initial_state, config=self.config, context={"model": self.model}):
                for node_name, node_output in event.items():
                    await self.handle_node_execution(node_name, node_output, log_feed, specialist_view)

                    # Check for approval gate interrupt (mission control)
                    if node_name == "approval_gate":
                        self.pending_action = node_output.get("pending_action", "Action requires approval")
                        approved = await self.show_approval_modal()

                        # Update state based on approval
                        if approved:
                            self.graph.update_state(
                                self.config,
                                {"operator_approved": True, "pending_action": None, "next_step": "tools"},
                                as_node="approval_gate",
                            )
                            log_feed.add_entry("APPROVAL", "Operator approved action")
                        else:
                            self.graph.update_state(
                                self.config,
                                {"operator_approved": False, "pending_action": None, "next_node": "end_node"},
                                as_node="approval_gate",
                            )
                            log_feed.add_entry("APPROVAL", "Operator rejected action")

                        # Resume graph
                        async for resume_event in self.graph.astream(None, config=self.config, context={"model": self.model}):
                            for rn, ro in resume_event.items():
                                await self.handle_node_execution(rn, ro, log_feed, specialist_view)

                    # Task 3 & 4: Check for human_review interrupt (certification HITL)
                    if node_name == "human_review":
                        project_desc = node_output.get("pending_action", "MAJOR project certification")
                        classification_reasoning = node_output.get("classification_reasoning", "Classified as MAJOR project")

                        # Fixed: Use async get_state (aget_state) to avoid InvalidStateError
                        final_state = await self.graph.aget_state(self.config)
                        project_description = ""
                        if final_state.values:
                            project_description = final_state.values.get("project_description", project_desc)

                        # Show CertificationApprovalModal (Task 3)
                        log_feed.add_entry("CERTIFICATION", "Human review required - opening approval modal")
                        certified = await self.show_certification_modal(project_description, classification_reasoning)

                        # Task 4: State Update and Resumption
                        if certified:
                            # CERTIFY: resume to cert_aerospace
                            self.graph.update_state(
                                self.config,
                                {"operator_approved": True, "pending_action": None},
                                as_node="human_review",
                            )
                            log_feed.add_entry("CERTIFICATION", "✓ Operator CERTIFIED the project")
                            self.update_graph_node("human_review", NodeStatus.COMPLETED)
                        else:
                            # REJECT: resume to end_node (skip certification)
                            self.graph.update_state(
                                self.config,
                                {"operator_approved": False, "pending_action": None, "next_step": "end"},
                                as_node="human_review",
                            )
                            log_feed.add_entry("CERTIFICATION", "✗ Operator REJECTED the project")
                            self.update_graph_node("human_review", NodeStatus.ERROR)

                        # Resume graph
                        async for resume_event in self.graph.astream(None, config=self.config, context={"model": self.model}):
                            for rn, ro in resume_event.items():
                                await self.handle_node_execution(rn, ro, log_feed, specialist_view)

        except Exception as e:
            log_feed.add_entry("ERROR", f"Execution error: {str(e)}")

        # Final state processing - Fixed: Use async get_state (aget_state)
        final_state = await self.graph.aget_state(self.config)
        if final_state.values:
            self.display_final_report(final_state.values, log_feed)
            # Update telemetry based on tool outputs
            self.update_telemetry_from_state(final_state.values, telemetry_panel)

        specialist_view.set_active("idle")

    async def handle_node_execution(
        self,
        node_name: str,
        node_output: dict,
        log_feed: LogFeed,
        specialist_view: SpecialistView,
    ) -> None:
        """Handle execution of a graph node."""
        # Update visual graph - mark previous nodes as completed, current as active
        self.update_graph_node(node_name, NodeStatus.ACTIVE)

        # Log node execution
        log_feed.add_entry(node_name.upper(), "Executing...")

        # Update specialist view based on node
        if "logistics" in node_name.lower() and "adapter" not in node_name.lower():
            specialist_view.set_active("logistics")
            log_feed.add_entry("LOGISTICS", "Specialist processing request")
        elif "systems" in node_name.lower() and "adapter" not in node_name.lower():
            specialist_view.set_active("systems")
            log_feed.add_entry("SYSTEMS", "Specialist processing request")
        elif "parallel" in node_name.lower():
            specialist_view.set_active("parallel")
            log_feed.add_entry("PARALLEL", "Both specialists active")
        elif node_name == "triage_node":
            log_feed.add_entry("TRIAGE", "Routing to specialist")
        elif node_name == "tools":
            log_feed.add_entry("TOOLS", "Executing telemetry tools")
        elif node_name == "approval_gate":
            log_feed.add_entry("APPROVAL", "Hit approval gate - awaiting operator input")
        elif node_name == "end_node":
            log_feed.add_entry("END", "Generating final report")
        elif "cert" in node_name.lower():
            specialist_view.set_active("certification")
            log_feed.add_entry("CERTIFICATION", f"Processing {node_name}")

        # Update next_step/next_node tracking
        if node_output and isinstance(node_output, dict):
            if "next_step" in node_output:
                next_step = node_output["next_step"]
                if next_step == "tools":
                    log_feed.add_entry(node_name.upper(), "Requesting tool execution")
                elif next_step == "approval_gate":
                    log_feed.add_entry(node_name.upper(), "Flagging critical anomaly")

        # Mark node as completed after a brief delay (visual effect)
        await asyncio.sleep(0.1)
        self.update_graph_node(node_name, NodeStatus.COMPLETED)

    async def show_approval_modal(self) -> bool:
        """Show approval modal and wait for user input."""
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("HITL", "Pausing for operator approval...")

        modal = ApprovalModal(self.pending_action or "Action requires approval")
        result = await self.push_screen_wait(modal)

        return result

    async def show_certification_modal(self, project_description: str, classification_reasoning: str) -> bool:
        """Show certification approval modal (Task 3) and wait for user input."""
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("CERTIFICATION", "Pausing for human review...")

        modal = CertificationApprovalModal(
            project_description or "Unknown project",
            classification_reasoning or "Classified as MAJOR project"
        )
        result = await self.push_screen_wait(modal)

        return result

    def display_final_report(self, state: MissionState, log_feed: LogFeed) -> None:
        """Display the final mission control report."""
        log_feed.add_raw("")
        log_feed.add_raw("=" * 60)
        log_feed.add_raw("MISSION CONTROL REPORT")
        log_feed.add_raw("=" * 60)

        log_feed.add_raw(f"\nOriginal Query: {state['user_query']}")

        log_feed.add_raw("\n--- Message Log ---")
        for msg in state.get("messages", []):
            if hasattr(msg, "content") and msg.content:
                role = getattr(msg, "type", "unknown")
                if role == "human":
                    role = "User"
                elif role == "ai":
                    role = "Assistant"
                elif role == "tool":
                    role = "Tool"
                log_feed.add_raw(f"[{role}] {msg.content}")

        tool_outputs = format_tool_outputs(state.get("messages", []))
        if tool_outputs:
            log_feed.add_raw("\n--- Telemetry Tools Output ---")
            for line in tool_outputs.split("\n"):
                log_feed.add_raw(line)

        log_feed.add_raw("\n--- Routing Decision ---")
        next_node = state.get("next_node", "unknown")
        if next_node == "logistics":
            log_feed.add_raw("Routed to: Logistics Specialist")
        elif next_node == "systems":
            log_feed.add_raw("Routed to: Systems Specialist")
        elif next_node == "tools":
            log_feed.add_raw("Routed to: Telemetry Tools")
        else:
            log_feed.add_raw("Routed to: Direct Response (no specialist needed)")

        if state.get("tool_failed"):
            log_feed.add_raw("\n--- Tool Status ---")
            log_feed.add_raw("Sensors Offline - Manual Override Needed")

        log_feed.add_raw("=" * 60)
        log_feed.add_raw("")

    def update_telemetry_from_state(self, state: MissionState, telemetry_panel: TelemetryPanel) -> None:
        """Update telemetry panel from final state."""
        # Parse tool outputs for telemetry data
        messages = state.get("messages", [])
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if "PSI" in content and "Stable" in content:
                    try:
                        pressure = int(content.split()[0])
                        telemetry_panel.update_telemetry(pressure=pressure)
                    except (ValueError, IndexError):
                        pass
                elif "Main:" in content and "RCS:" in content:
                    try:
                        parts = content.split()
                        main = int(parts[1].rstrip("%,"))
                        rcs = int(parts[4].rstrip("%,"))
                        telemetry_panel.update_telemetry(fuel_main=main, fuel_rcs=rcs)
                    except (ValueError, IndexError):
                        pass

    def action_save_session(self) -> None:
        """Save current session."""
        log_feed = self.query_one(LogFeed)
        log_feed.add_entry("SYSTEM", f"Session {self.session_id} saved")

    def action_show_help(self) -> None:
        """Show help information with dynamic examples (Step 3)."""
        log_feed = self.query_one(LogFeed)
        log_feed.add_raw("")
        log_feed.add_raw("╔══════════════════════════════════════════════════════════════════╗")
        log_feed.add_raw("║  HELP - TINYGRAPH MISSION CONTROL                                ║")
        log_feed.add_raw("╠══════════════════════════════════════════════════════════════════╣")
        log_feed.add_raw("║  KEYBOARD SHORTCUTS:                                             ║")
        log_feed.add_raw("║  ├─ Ctrl+C: Quit application                                     ║")
        log_feed.add_raw("║  ├─ Ctrl+S: Save current session                                 ║")
        log_feed.add_raw("║  ├─ Ctrl+R: Reset graph visual                                   ║")
        log_feed.add_raw("║  ├─ Ctrl+L: List past certifications                             ║")
        log_feed.add_raw("║  ├─ F1: Show this help                                           ║")
        log_feed.add_raw("║  ├─ [1]: Switch to Mission Control Graph path                    ║")
        log_feed.add_raw("║  └─ [2]: Switch to Certification path                            ║")
        log_feed.add_raw("╠══════════════════════════════════════════════════════════════════╣")
        log_feed.add_raw("║  💡 QUICK START EXAMPLES (copy-paste):                           ║")
        log_feed.add_raw("║  🚀 MISSION:                                                     ║")
        for ex in MISSION_EXAMPLES:
            log_feed.add_raw(f"║    • {ex[:58]:<58} ║")
        log_feed.add_raw("║  📋 CERTIFICATION:                                               ║")
        for ex in CERTIFICATION_EXAMPLES:
            log_feed.add_raw(f"║    • {ex[:58]:<58} ║")
        log_feed.add_raw("╠══════════════════════════════════════════════════════════════════╣")
        log_feed.add_raw("║  OPERATION MODES:                                                ║")
        log_feed.add_raw("║  ├─ 🚀 MISSION CONTROL: Triage queries to specialists            ║")
        log_feed.add_raw("║  │  • Logistics: Fuel, Cargo, Resupply, Docking                   ║")
        log_feed.add_raw("║  │  • Systems: Propulsion, Navigation, Life Support               ║")
        log_feed.add_raw("║  │  • Emergency: Auto-routes on high pressure                     ║")
        log_feed.add_raw("║  └─ 📋 CERTIFICATION: Classify & certify projects               ║")
        log_feed.add_raw("║     • PRC: Minor/Major classification                             ║")
        log_feed.add_raw("║     • CEP: Full certification programme (Major only)             ║")
        log_feed.add_raw("╠══════════════════════════════════════════════════════════════════╣")
        log_feed.add_raw("║  GRAPH NODES:                                                    ║")
        log_feed.add_raw("║  ● Active  ✓ Complete  ✗ Error  ○ Pending                        ║")
        log_feed.add_raw("╚══════════════════════════════════════════════════════════════════╝")
        log_feed.add_raw("")


# ============================================================================
# Main Entry Point
# ============================================================================

async def run_tui(model_path: str = "D:\\models\\qwen.gguf", db_path: str = "mission_sessions.sqlite", session_id: str | None = None) -> int:
    """Run the TUI dashboard with AsyncSqliteSaver.

    Task 1: Fix Persistence Implementation
    Establishes aiosqlite connection using 'async with' and initializes AsyncSqliteSaver.
    """
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully.\n")

    session_id = session_id or str(uuid.uuid4())

    # Task 1: Establish aiosqlite connection first, then initialize AsyncSqliteSaver
    async with aiosqlite.connect(db_path, check_same_thread=False) as conn:
        # Ensure certifications table exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS certifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                project_description TEXT,
                classification TEXT,
                status TEXT,
                stage TEXT,
                report TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()

        # Initialize AsyncSqliteSaver with the connection
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver(conn)

        app = GroundStationDashboard(model, checkpointer, session_id)
        await app.run_async()

    return 0


# ============================================================================
# Standalone Graph Visualization (No Model Required)
# ============================================================================

def visualize_graph() -> None:
    """Print a visual representation of the graph to console."""
    console = Console()

    # Create main graph tree
    tree = Tree("🚀 [bold cyan]TINYGRAPH - Mission Control Graph[/bold cyan]")

    # Triage node
    triage = tree.add("🔍 [cyan]TRIAGE_NODE[/cyan] [dim](Router)[/dim]")
    triage.add("Analyzes query & routes to specialist")

    # Logistics branch
    logistics = triage.add("📥 [green]LOGISTICS ADAPTER_IN[/green] [dim](Adapter)[/dim]")
    logistics_sub = logistics.add("🚀 [green]LOGISTICS SPECIALIST[/green] [dim](Specialist)[/dim]")
    logistics_sub.add("Fuel, Cargo, Resupply, Docking")
    logistics_out = logistics.add("📤 [green]LOGISTICS ADAPTER_OUT[/green] [dim](Adapter)[/dim]")

    # Systems branch
    systems = triage.add("📥 [yellow]SYSTEMS ADAPTER_IN[/yellow] [dim](Adapter)[/dim]")
    systems_sub = systems.add("⚙️ [yellow]SYSTEMS SPECIALIST[/yellow] [dim](Specialist)[/dim]")
    systems_sub.add("Propulsion, Navigation, Life Support")
    systems_out = systems.add("📤 [yellow]SYSTEMS ADAPTER_OUT[/yellow] [dim](Adapter)[/dim]")

    # Parallel branch
    parallel = triage.add("🔀 [magenta]PARALLEL HANDLER[/magenta] [dim](Router)[/dim]")
    parallel.add("Routes to both specialists")
    merger = parallel.add("🔗 [magenta]PARALLEL MERGER[/magenta] [dim](Merger)[/dim]")

    # Tools & Approval
    for branch in [logistics_out, systems_out, merger]:
        tools = branch.add("📊 [blue]TOOLS[/blue] [dim](Executor)[/dim]")
        tools.add("read_pressure, check_fuel, log_anomaly")
        approval = branch.add("🛡️ [red]APPROVAL GATE[/red] [dim](HITL)[/dim]")
        approval.add("Human-in-the-loop approval")

    # End node
    tree.add("🏁 [bright_cyan]END_NODE[/bright_cyan] [dim](Terminal)[/dim]").add("Generates final report")

    console.print(tree)

    # Create certification tree
    console.print("\n")
    cert_tree = Tree("📋 [bold bright_magenta]PROJECT CLASSIFICATION & CERTIFICATION[/bold bright_magenta]")
    cert_triage = cert_tree.add("📋 [bright_magenta]CERT TRIAGE[/bright_magenta] [dim](Router)[/dim]")
    cert_triage.add("Classify project type & requirements")

    aero = cert_triage.add("🛸 [bright_blue]AEROSPACE CERT[/bright_blue] [dim](Certifier)[/dim]")
    aero.add("Aerospace compliance checks")

    mission = cert_triage.add("🎯 [bright_green]MISSION CERT[/bright_green] [dim](Certifier)[/dim]")
    mission.add("Mission requirements validation")

    safety = cert_triage.add("⚠️ [bright_red]SAFETY CERT[/bright_red] [dim](Certifier)[/dim]")
    safety.add("Safety & risk assessment")

    cert_tree.add("✅ [bright_cyan]FINAL CERT[/bright_cyan] [dim](Terminal)[/dim]").add("Certification report & approval")

    console.print(cert_tree)

    # Legend
    console.print("\n")
    legend = Table(title="Node Types Legend", show_header=True, header_style="bold")
    legend.add_column("Type", style="cyan")
    legend.add_column("Description", style="white")
    legend.add_column("Color", style="dim")

    legend.add_row("Router", "Routes to next node based on conditions", "cyan/magenta")
    legend.add_row("Adapter", "Maps state between parent and subgraph", "green/yellow")
    legend.add_row("Specialist", "Domain expert (Logistics/Systems)", "green/yellow")
    legend.add_row("Executor", "Runs tools (Telemetry, Sensors)", "blue")
    legend.add_row("HITL", "Human-in-the-loop (Approval)", "red")
    legend.add_row("Terminal", "End node (Final report)", "bright_cyan")
    legend.add_row("Certifier", "Certification checks", "bright colors")

    console.print(legend)

    console.print("\n[bold green]✓[/bold green] Graph visualization complete!")
    console.print("[dim]Run the TUI with: python tui.py[/dim]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        visualize_graph()
    else:
        sys.exit(asyncio.run(run_tui()))
