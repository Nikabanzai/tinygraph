"""Logistics subgraph definition."""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from logistics.state import LogisticsState


class LogisticsContext(TypedDict):
    """Runtime context passed from the parent graph."""

    model: object


def build_tool_call(tool_name: str, args: dict) -> dict:
    """Create a tool call payload for the ToolNode."""
    return {
        "name": tool_name,
        "args": args,
        "id": args.get("_tool_call_id", ""),
        "type": "tool_call",
    }


def format_tool_outputs(messages: list) -> str:
    """Format tool outputs from ToolMessage entries."""
    outputs = []
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            outputs.append(f"- {msg.name}: {msg.content}")
    return "\n".join(outputs)


def extract_json(text: str) -> dict:
    """Extract JSON payload from an LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def is_critical_anomaly(tool_call: dict) -> bool:
    """Check if a tool call is requesting a critical anomaly log."""
    if tool_call.get("name") != "log_anomaly":
        return False
    severity = tool_call.get("args", {}).get("severity")
    return isinstance(severity, str) and severity.lower() == "critical"


def logistics_node(state: LogisticsState, *, runtime) -> LogisticsState:
    """Logistics specialist node handles fuel, cargo, resupply, and docking."""
    query = state["user_query"]
    status_msg = f"[LOGISTICS] Processing logistics request: {query}"
    print(status_msg)

    model = runtime.context["model"] if runtime.context else None

    if model is None:
        message = AIMessage(content="Model unavailable - unable to process logistics request")
        return {
            "messages": [message],
            "next_step": "end_node",
            "tool_failed": True,
            "tool_error": "Model unavailable - unable to process logistics request",
        }

    if state.get("tool_failed"):
        message = AIMessage(content="Sensors Offline - Manual Override Needed")
        return {
            "messages": [message],
            "next_step": "end_node",
            "tool_failed": True,
            "tool_error": "Sensors Offline - Manual Override Needed",
        }

    prompt = f"""You are the logistics specialist for mission control.

User Query: {query}

Available tools:
- read_pressure(id: str)
- check_fuel()
- log_anomaly(details: str)

Decide if a tool is required. Respond with JSON only.
If a tool is needed, use:
{{"action": "tool", "tool_name": "check_fuel", "tool_args": {{}}, "report": ""}}
If no tool is needed, use:
{{"action": "report", "tool_name": "", "tool_args": {{}}, "report": "<your report>"}}
"""

    response = model(prompt, max_tokens=200, temperature=0.0, stop=["\n\n"])
    content = response["choices"][0]["text"].strip()
    data = extract_json(content)

    if data.get("action") == "tool":
        tool_name = data.get("tool_name", "")
        tool_args = data.get("tool_args", {})
        tool_call = build_tool_call(tool_name, tool_args)
        message = AIMessage(content="", tool_calls=[tool_call])
        next_step = "tools"
        pending_action = None
        if is_critical_anomaly(tool_call):
            next_step = "approval_gate"
            pending_action = "Flagging CRITICAL anomaly"
        return {
            "messages": [message],
            "next_step": next_step,
            "last_tool_caller": "logistics",
            "tool_failed": False,
            "tool_error": "",
            "pending_action": pending_action,
            "operator_approved": None,
        }

    report = data.get("report") or f"Logistics specialist handling: {query}"
    telemetry = format_tool_outputs(state.get("messages", []))
    if telemetry:
        report = f"{report}\n\nTelemetry Results:\n{telemetry}"
    message = AIMessage(content=report)
    return {
        "messages": [message],
        "next_step": "end_node",
        "tool_attempts": 0,
        "tool_failed": False,
        "tool_error": "",
    }


def build_logistics_graph(checkpointer: Checkpointer | None = None):
    """Build and compile the logistics subgraph."""
    builder = StateGraph(LogisticsState, context_schema=LogisticsContext)
    builder.add_node("logistics", logistics_node)
    builder.add_edge(START, "logistics")
    builder.add_edge("logistics", END)
    return builder.compile(checkpointer=checkpointer)


logistics_graph = build_logistics_graph()
