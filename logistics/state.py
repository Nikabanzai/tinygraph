"""State definitions for the logistics subgraph."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class LogisticsState(TypedDict):
    """State for the logistics specialist subgraph."""

    messages: Annotated[list, add_messages]
    user_query: str
    next_step: str
    tool_attempts: int
    tool_failed: bool
    tool_error: str
    last_tool_caller: str
    operator_approved: Optional[bool]
    pending_action: Optional[str]
