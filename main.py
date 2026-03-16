"""Aerospace Mission Control Agent using LangGraph."""

import argparse
import sys
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llama_cpp import Llama

import sqlite3
import uuid

from logistics.graph import build_logistics_graph
from logistics.state import LogisticsState
from systems.graph import build_systems_graph
from systems.state import SystemsState
from certification.graph import build_certification_graph
from certification.state import CertificationState
from tools import check_fuel, log_anomaly, read_pressure

# High pressure threshold for emergency routing
HIGH_PRESSURE_THRESHOLD = 500
CRITICAL_PRESSURE_THRESHOLD = 700


class MissionState(TypedDict):
    """State for the mission control agent."""

    messages: Annotated[list, add_messages]
    user_query: str
    next_node: str
    next_step: str
    tool_attempts: int
    tool_failed: bool
    tool_error: str
    last_tool_caller: str
    operator_approved: Optional[bool]
    pending_action: Optional[str]


def load_model(model_path: str) -> Llama:
    """Load the GGUF model from the given path."""
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )


def build_checkpointer(db_path: str) -> SqliteSaver:
    """Create a synchronous SQLite checkpointer for session persistence (main.py sync usage)."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Ensure certifications table exists
    cursor = conn.cursor()
    cursor.execute("""
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
    conn.commit()
    return SqliteSaver(conn)


async def build_async_checkpointer(db_path: str) -> AsyncSqliteSaver:
    """Create an async SQLite checkpointer for session persistence (TUI async usage).

    Step 1: Migration to AsyncSqliteSaver
    Uses async with context managers for the checkpointer connection.
    """
    # Ensure certifications table exists (sync for setup)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
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
    conn.commit()
    conn.close()

    # Create async checkpointer
    async_conn = await aiosqlite.connect(db_path, check_same_thread=False)
    return AsyncSqliteSaver(async_conn)


def save_certification(
    conn: sqlite3.Connection,
    session_id: str,
    project_description: str,
    classification: str,
    status: str,
    stage: str,
    report: str,
) -> int:
    """Save a certification record to SQLite.

    Task 3: Certification Persistence
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO certifications (session_id, project_description, classification, status, stage, report)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, project_description, classification, status, stage, report))
    conn.commit()
    return cursor.lastrowid


def get_certifications(conn: sqlite3.Connection, session_id: str | None = None) -> list[dict]:
    """Retrieve certification records from SQLite.

    Task 3: Certification Persistence
    Returns list of certification dicts.
    """
    cursor = conn.cursor()
    if session_id:
        cursor.execute("""
            SELECT id, session_id, project_description, classification, status, stage, report, created_at
            FROM certifications
            WHERE session_id = ?
            ORDER BY created_at DESC
        """, (session_id,))
    else:
        cursor.execute("""
            SELECT id, session_id, project_description, classification, status, stage, report, created_at
            FROM certifications
            ORDER BY created_at DESC
            LIMIT 20
        """)
    rows = cursor.fetchall()
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "project_description": row[2],
            "classification": row[3],
            "status": row[4],
            "stage": row[5],
            "report": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]


def format_tool_outputs(messages: list) -> str:
    """Format tool outputs from ToolMessage entries."""
    outputs = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            outputs.append(f"- {msg.name}: {msg.content}")
    return "\n".join(outputs)


def extract_tool_calls(messages: list) -> list[dict]:
    """Extract the latest tool calls from the message history."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return list(msg.tool_calls)
    return []


def triage_node(state: MissionState, model: Llama) -> MissionState:
    """Triage node reads the query and decides where to route it."""
    query = state["user_query"]
    normalized_query = query.lower()

    # Heuristic keyword routing to avoid LLM ambiguity
    logistics_keywords = {"fuel", "cargo", "resupply", "docking"}
    systems_keywords = {
        "propulsion",
        "navigation",
        "life support",
        "reactor",
        "propulsion failure",
        "engine failure",
        "thruster failure",
        "guidance failure",
        "power failure",
    }

    # Check for composite queries (mentions both domains)
    has_logistics = any(keyword in normalized_query for keyword in logistics_keywords)
    has_systems = any(keyword in normalized_query for keyword in systems_keywords)

    if has_logistics and has_systems:
        # Composite query - will be handled with parallel execution
        return {"next_node": "parallel"}

    if has_logistics:
        return {"next_node": "logistics_adapter_in"}

    if has_systems:
        return {"next_node": "systems_adapter_in"}

    # Build prompt for LLM triage decision
    prompt = f"""You are a mission control triage system. Analyze the following query and decide which specialist should handle it.

Query: {query}

Routing rules:
- If the query is about fuel, cargo, resupply, docking, or payload logistics, route to logistics
- If the query is about propulsion, navigation, life support, reactor, engines, power systems, or failure modes, route to systems
- For everything else, route to end_node

Respond with ONLY one of these exact words: logistics_adapter_in, systems_adapter_in, or end_node"""

    # Query the LLM
    response = model(
        prompt,
        max_tokens=10,
        temperature=0.0,
        stop=["\n"],
    )

    # Extract the decision
    decision = response["choices"][0]["text"].strip().lower()

    # Validate and default
    if "logistics" in decision:
        return {"next_node": "logistics_adapter_in"}
    if "systems" in decision:
        return {"next_node": "systems_adapter_in"}

    return {"next_node": "end_node"}


def triage_router(
    state: MissionState,
) -> Literal["logistics_adapter_in", "systems_adapter_in", "parallel", "end_node"]:
    """Conditional edge router based on triage decision."""
    return state["next_node"]


def parallel_handler(state: MissionState, model: Llama) -> MissionState:
    """Handle composite queries by routing to both specialists."""
    # Store that this is a parallel execution
    return {
        "next_node": "parallel",
        "next_step": "parallel",
        "messages": state.get("messages", []),
    }


def parallel_merger(state: MissionState) -> MissionState:
    """Merge results from parallel specialist execution."""
    messages = state.get("messages", [])
    # Add a merged summary message
    merge_msg = AIMessage(content="[MERGED REPORT] Combined output from Logistics and Systems specialists")
    return {
        "messages": messages + [merge_msg],
        "next_node": "end_node",
        "next_step": "end_node",
    }


def logistics_adapter_in(state: MissionState) -> LogisticsState:
    """Map parent state to logistics subgraph state."""
    return {
        "messages": state["messages"],
        "user_query": state["user_query"],
        "next_step": "logistics",
        "tool_attempts": state.get("tool_attempts", 0),
        "tool_failed": state.get("tool_failed", False),
        "tool_error": state.get("tool_error", ""),
        "last_tool_caller": state.get("last_tool_caller", ""),
        "operator_approved": state.get("operator_approved"),
        "pending_action": state.get("pending_action"),
    }


def logistics_adapter_out(state: LogisticsState) -> MissionState:
    """Map logistics subgraph output back to parent state."""
    next_step = state.get("next_step", "")
    return {
        "messages": state.get("messages", []),
        "next_node": next_step,
        "next_step": next_step,
        "tool_attempts": state.get("tool_attempts", 0),
        "tool_failed": state.get("tool_failed", False),
        "tool_error": state.get("tool_error", ""),
        "last_tool_caller": "logistics",
        "operator_approved": state.get("operator_approved"),
        "pending_action": state.get("pending_action"),
    }


def systems_adapter_in(state: MissionState) -> SystemsState:
    """Map parent state to systems subgraph state."""
    return {
        "messages": state["messages"],
        "user_query": state["user_query"],
        "next_step": "systems",
        "tool_attempts": state.get("tool_attempts", 0),
        "tool_failed": state.get("tool_failed", False),
        "tool_error": state.get("tool_error", ""),
        "last_tool_caller": state.get("last_tool_caller", ""),
        "operator_approved": state.get("operator_approved"),
        "pending_action": state.get("pending_action"),
    }


def systems_adapter_out(state: SystemsState) -> MissionState:
    """Map systems subgraph output back to parent state."""
    next_step = state.get("next_step", "")
    return {
        "messages": state.get("messages", []),
        "next_node": next_step,
        "next_step": next_step,
        "tool_attempts": state.get("tool_attempts", 0),
        "tool_failed": state.get("tool_failed", False),
        "tool_error": state.get("tool_error", ""),
        "last_tool_caller": "systems",
        "operator_approved": state.get("operator_approved"),
        "pending_action": state.get("pending_action"),
    }


def end_node(state: MissionState) -> MissionState:
    """End node prints the full report."""
    print("\n" + "=" * 60)
    print("MISSION CONTROL REPORT")
    print("=" * 60)

    print(f"\nOriginal Query: {state['user_query']}")

    print("\n--- Message Log ---")
    for msg in state["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = getattr(msg, "type", "unknown")
            if role == "human":
                role = "User"
            elif role == "ai":
                role = "Assistant"
            elif role == "tool":
                role = "Tool"
            print(f"[{role}] {msg.content}")

    tool_outputs = format_tool_outputs(state["messages"])
    if tool_outputs:
        print("\n--- Telemetry Tools Output ---")
        print(tool_outputs)

    print("\n--- Routing Decision ---")
    next_node = state.get("next_node", "unknown")
    if next_node == "logistics":
        print("Routed to: Logistics Specialist")
    elif next_node == "systems":
        print("Routed to: Systems Specialist")
    elif next_node == "tools":
        print("Routed to: Telemetry Tools")
    else:
        print("Routed to: Direct Response (no specialist needed)")

    if state.get("tool_failed"):
        print("\n--- Tool Status ---")
        print("Sensors Offline - Manual Override Needed")

    print("=" * 60 + "\n")

    return {}


def approval_gate(state: MissionState) -> MissionState:
    """Approval gate for critical anomaly logging."""
    return {
        "pending_action": "Flagging CRITICAL anomaly",
        "next_step": "approval_gate",
    }


def parse_pressure_value(content: str) -> int | None:
    """Parse pressure value from tool output content."""
    try:
        # Expected format: "340 PSI - Stable" or similar
        parts = content.split()
        for i, part in enumerate(parts):
            if part.upper() == "PSI" and i > 0:
                return int(parts[i - 1])
        # Try direct parse
        for word in content.split():
            if word.isdigit():
                return int(word)
    except (ValueError, IndexError):
        pass
    return None


def check_emergency_condition(messages: list) -> dict | None:
    """Check if telemetry results trigger emergency routing.

    Returns emergency info dict if emergency detected, None otherwise.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "read_pressure":
            pressure = parse_pressure_value(msg.content)
            if pressure is not None:
                if pressure >= CRITICAL_PRESSURE_THRESHOLD:
                    return {
                        "level": "CRITICAL",
                        "pressure": pressure,
                        "action": "emergency_systems_check",
                        "message": f"CRITICAL PRESSURE: {pressure} PSI - Emergency Systems Check Required!",
                    }
                elif pressure >= HIGH_PRESSURE_THRESHOLD:
                    return {
                        "level": "HIGH",
                        "pressure": pressure,
                        "action": "systems_check",
                        "message": f"HIGH PRESSURE: {pressure} PSI - Systems Check Recommended",
                    }
    return None


def tools_node(state: MissionState, tool_node: ToolNode) -> MissionState:
    """Execute telemetry tools with retry handling and emergency detection."""
    tool_calls = extract_tool_calls(state.get("messages", []))
    if not tool_calls:
        return {"next_node": state.get("last_tool_caller", "end_node"), "next_step": state.get("next_step", "")}

    attempts = state.get("tool_attempts", 0)
    collected_messages: list[ToolMessage] = []

    while attempts < 2:
        simulate_failure = any(
            call.get("args", {}).get("simulate_fail") for call in tool_calls
        )
        if simulate_failure:
            attempts += 1
            collected_messages.append(
                ToolMessage(
                    content="Simulated tool failure",
                    name=tool_calls[0]["name"],
                    tool_call_id=tool_calls[0]["id"],
                    status="error",
                )
            )
        else:
            result = tool_node.invoke(state)
            tool_messages = result["messages"] if isinstance(result, dict) else result
            collected_messages.extend(tool_messages)
            has_error = any(
                isinstance(msg, ToolMessage) and msg.status == "error"
                for msg in tool_messages
            )
            if not has_error:
                # Check for emergency conditions (Tool-Triggered Routing - Task 2)
                emergency = check_emergency_condition(collected_messages)
                if emergency:
                    emergency_msg = AIMessage(content=f"[EMERGENCY] {emergency['message']}")
                    collected_messages.append(emergency_msg)

                    # Route to emergency systems check for critical pressure
                    if emergency["level"] == "CRITICAL":
                        return {
                            "messages": collected_messages,
                            "tool_attempts": 0,
                            "tool_failed": False,
                            "tool_error": "",
                            "next_node": "emergency_systems_check",
                            "next_step": "emergency_systems_check",
                            "pending_action": emergency["message"],
                        }
                    # For high (not critical), flag for attention but continue
                    elif emergency["level"] == "HIGH":
                        warning_msg = AIMessage(content=f"[WARNING] {emergency['message']}")
                        collected_messages.append(warning_msg)

                return {
                    "messages": collected_messages,
                    "tool_attempts": 0,
                    "tool_failed": False,
                    "tool_error": "",
                    "next_node": state.get("last_tool_caller", "end_node"),
                    "next_step": state.get("next_step", ""),
                }
            attempts += 1

        if attempts >= 2:
            break

    error_message = "Sensors Offline - Manual Override Needed"
    collected_messages.append(
        ToolMessage(
            content=error_message,
            name=tool_calls[0]["name"],
            tool_call_id=tool_calls[0]["id"],
            status="error",
        )
    )
    collected_messages.append(AIMessage(content=error_message))
    return {
        "messages": collected_messages,
        "tool_attempts": attempts,
        "tool_failed": True,
        "tool_error": error_message,
        "next_node": "end_node",
        "next_step": state.get("next_step", ""),
    }


def emergency_systems_check(state: MissionState) -> MissionState:
    """Emergency systems check node triggered by high pressure telemetry.

    This is part of Tool-Triggered Routing (Task 2).
    """
    messages = state.get("messages", [])

    # Analyze the emergency condition
    emergency_report = """
[EMERGENCY SYSTEMS CHECK]

⚠️  HIGH PRESSURE CONDITION DETECTED

Automatic Systems Check Initiated:
1. ✓ Pressure relief valves - CHECKING
2. ✓ Containment systems - VERIFYING
3. ✓ Backup systems - STANDBY
4. ✓ Crew notification - ALERTING

Recommended Actions:
- Reduce system pressure immediately
- Activate backup containment if pressure exceeds 800 PSI
- Prepare for potential emergency shutdown

Human operator intervention recommended for critical decisions.
"""

    emergency_msg = AIMessage(content=emergency_report)
    messages = messages + [emergency_msg]

    return {
        "messages": messages,
        "next_node": "end_node",
        "next_step": "end_node",
    }


def tool_router(state: MissionState) -> Literal["tools", "approval_gate", "end_node"]:
    """Route to tools, approval gate, or end based on specialist decision."""
    next_node = state.get("next_node")
    if next_node == "approval_gate":
        return "approval_gate"
    if next_node == "tools":
        return "tools"
    return "end_node"


def approval_router(state: MissionState) -> Literal["tools", "end_node"]:
    """Route from approval gate based on operator decision."""
    if state.get("operator_approved") is False:
        return "end_node"
    return "tools"


def tools_return_router(
    state: MissionState,
) -> Literal["logistics", "systems", "parallel", "end_node"]:
    """Route back to the requesting specialist or end after tools."""
    if state.get("tool_failed"):
        return "end_node"
    return state.get("last_tool_caller", "end_node")


def certification_adapter_in(state: MissionState) -> CertificationState:
    """Map parent state to certification subgraph state."""
    # Extract project description from user query
    project_description = state.get("user_query", "")
    return {
        "messages": state.get("messages", []),
        "user_query": state.get("user_query", ""),
        "project_description": project_description,
        "project_classification": "",
        "classification_reasoning": "",
        "certification_status": "Pending",
        "certification_stage": "PRC - Starting",
        "aerospace_compliance": None,
        "mission_requirements": None,
        "safety_assessment": None,
        "certification_report": "",
        "next_step": "",
        "tool_attempts": 0,
        "tool_failed": False,
        "tool_error": "",
        "last_tool_caller": "",
        "operator_approved": None,
        "pending_action": None,
    }


def certification_adapter_out(state: CertificationState) -> MissionState:
    """Map certification subgraph output back to parent state."""
    certification_status = state.get("certification_status", "Unknown")
    certification_report = state.get("certification_report", "")
    project_classification = state.get("project_classification", "Unknown")

    messages = state.get("messages", [])
    summary = AIMessage(
        content=f"[CERTIFICATION COMPLETE] Status: {certification_status} | Classification: {project_classification}"
    )

    return {
        "messages": messages + [summary],
        "next_node": "end_node",
        "next_step": "end_node",
        "tool_attempts": 0,
        "tool_failed": False,
        "tool_error": "",
        "last_tool_caller": "certification",
    }


def build_graph(model: Llama, checkpointer: SqliteSaver | AsyncSqliteSaver, interrupt_before: list[str] | None = None) -> StateGraph:
    """Build and compile the mission control graph.

    Includes:
    - Mission Control path (Logistics, Systems specialists)
    - Emergency Systems Check (Tool-Triggered Routing - Task 2)
    - Certification path (PRC + CEP - Task 1)
    """
    builder = StateGraph(MissionState)

    tool_node = ToolNode(
        [read_pressure, check_fuel, log_anomaly]
    )

    logistics_subgraph = build_logistics_graph(checkpointer=True)
    systems_subgraph = build_systems_graph(checkpointer=True)
    certification_subgraph = build_certification_graph(checkpointer=True)

    # Add nodes
    builder.add_node("triage_node", lambda state: triage_node(state, model))
    builder.add_node("parallel_handler", lambda state: parallel_handler(state, model))
    builder.add_node("parallel_merger", parallel_merger)
    builder.add_node("logistics_adapter_in", logistics_adapter_in)
    builder.add_node(
        "logistics",
        logistics_subgraph,
        input_schema=LogisticsState,
    )
    builder.add_node("logistics_adapter_out", logistics_adapter_out)
    builder.add_node("systems_adapter_in", systems_adapter_in)
    builder.add_node(
        "systems",
        systems_subgraph,
        input_schema=SystemsState,
    )
    builder.add_node("systems_adapter_out", systems_adapter_out)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("tools", lambda state: tools_node(state, tool_node))
    builder.add_node("emergency_systems_check", emergency_systems_check)
    builder.add_node("certification_adapter_in", certification_adapter_in)
    builder.add_node(
        "certification",
        certification_subgraph,
        input_schema=CertificationState,
    )
    builder.add_node("certification_adapter_out", certification_adapter_out)
    builder.add_node("end_node", end_node)

    # Add edges
    builder.add_edge(START, "triage_node")
    builder.add_conditional_edges(
        "triage_node",
        triage_router,
        {
            "logistics_adapter_in": "logistics_adapter_in",
            "systems_adapter_in": "systems_adapter_in",
            "parallel": "parallel_handler",
            "end_node": "end_node",
        },
    )

    # Parallel execution path
    builder.add_edge("parallel_handler", "logistics_adapter_in")
    builder.add_edge("logistics_adapter_out", "systems_adapter_in")
    builder.add_edge("systems_adapter_out", "parallel_merger")
    builder.add_edge("parallel_merger", "end_node")

    # Single specialist paths
    builder.add_edge("logistics_adapter_in", "logistics")
    builder.add_edge("logistics", "logistics_adapter_out")
    builder.add_edge("systems_adapter_in", "systems")
    builder.add_edge("systems", "systems_adapter_out")

    builder.add_conditional_edges(
        "logistics_adapter_out",
        tool_router,
        {
            "tools": "tools",
            "approval_gate": "approval_gate",
            "end_node": "end_node",
        },
    )
    builder.add_conditional_edges(
        "systems_adapter_out",
        tool_router,
        {
            "tools": "tools",
            "approval_gate": "approval_gate",
            "end_node": "end_node",
        },
    )

    builder.add_conditional_edges(
        "approval_gate",
        approval_router,
        {"tools": "tools", "end_node": "end_node"},
    )

    builder.add_conditional_edges(
        "tools",
        tools_return_router,
        {
            "logistics": "logistics_adapter_in",
            "systems": "systems_adapter_in",
            "parallel": "parallel_merger",
            "end_node": "end_node",
            "emergency_systems_check": "emergency_systems_check",
        },
    )

    # Emergency systems check routes to end
    builder.add_edge("emergency_systems_check", "end_node")

    # Certification path (can be triggered via certification_adapter_in)
    builder.add_edge("certification_adapter_in", "certification")
    builder.add_edge("certification", "certification_adapter_out")
    builder.add_edge("certification_adapter_out", "end_node")

    builder.add_edge("end_node", END)

    interrupt_nodes = interrupt_before if interrupt_before is not None else ["approval_gate"]
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )


def main():
    """Main entry point for the mission control agent."""
    parser = argparse.ArgumentParser(
        description="Aerospace Mission Control Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="D:\\models\\qwen.gguf",
        help="Path to the GGUF model file (default: D:\\models\\qwen.gguf)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="User query to process (if not provided, reads from stdin)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for SQLite persistence (auto-generated if omitted)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="mission_sessions.sqlite",
        help="SQLite DB file for session persistence",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)
    print("Model loaded successfully.\n")

    # Configure session persistence
    session_id = args.session_id or str(uuid.uuid4())
    checkpointer = build_checkpointer(args.db_path)
    config = {"configurable": {"thread_id": session_id}}

    # Build the graph
    graph = build_graph(model, checkpointer)

    # Get the query
    if args.query:
        user_query = args.query
    else:
        user_query = input("Enter your mission control query: ")

    # Initialize or resume state
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "next_node": "",
        "next_step": "",
        "tool_attempts": 0,
        "tool_failed": False,
        "tool_error": "",
        "last_tool_caller": "",
        "operator_approved": None,
        "pending_action": None,
    }

    # Run the graph
    print(f"\nProcessing query: {user_query}\n")
    graph.invoke(initial_state, config=config, context={"model": model})

    while True:
        snapshot = graph.get_state(config)
        if "approval_gate" not in snapshot.next:
            break

        pending_action = snapshot.values.get("pending_action")
        if pending_action:
            print(f"\nPending Action: {pending_action}")
        else:
            print("\nPending Action: Approval required")

        decision = input("[A]pprove or [R]eject? ").strip().lower()
        if decision.startswith("a"):
            graph.update_state(
                config,
                {"operator_approved": True, "pending_action": None},
                as_node="approval_gate",
            )
            graph.invoke(None, config=config, context={"model": model})
        elif decision.startswith("r"):
            graph.update_state(
                config,
                {
                    "operator_approved": False,
                    "pending_action": None,
                    "next_node": "end_node",
                },
                as_node="approval_gate",
            )
            graph.invoke(None, config=config, context={"model": model})
            break
        else:
            print("Invalid input. Please enter A or R.")

    print(f"Session ID: {session_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
