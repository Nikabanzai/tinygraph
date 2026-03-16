"""State definitions for the certification subgraph."""

from typing import Annotated, Optional, TypedDict, Literal

from langgraph.graph.message import add_messages


class CertificationState(TypedDict):
    """State for the certification subgraph."""

    messages: Annotated[list, add_messages]
    user_query: str
    project_description: str
    project_classification: str  # "Minor" or "Major"
    classification_reasoning: str
    certification_status: str  # "Passed", "Failed", "Pending"
    certification_stage: str  # Current stage in CEP
    aerospace_compliance: Optional[dict]
    mission_requirements: Optional[dict]
    safety_assessment: Optional[dict]
    certification_report: str
    next_step: str
    tool_attempts: int
    tool_failed: bool
    tool_error: str
    last_tool_caller: str
    operator_approved: Optional[bool]
    pending_action: Optional[str]


# Certification stages for Major projects (CEP)
CERTIFICATION_STAGES = {
    "cert_triage": "PRC - Project Classification",
    "cert_aerospace": "CEP - Aerospace Compliance",
    "cert_mission": "CEP - Mission Requirements",
    "cert_safety": "CEP - Safety Assessment",
    "cert_final": "CEP - Final Certification",
}


# Major project criteria keywords (used as hints, LLM makes final decision)
MAJOR_CRITERIA_KEYWORDS = {
    "novel_technology": [
        "novel", "new technology", "unproven", "experimental", "prototype",
        "first-of-its-kind", "innovative", "cutting-edge", "breakthrough",
    ],
    "high_complexity": [
        "complex", "high complexity", "multiple systems", "integrated",
        "interdependent", "sophisticated", "advanced", "multi-stage",
    ],
    "critical_safety": [
        "safety-critical", "human-rated", "crew", "passenger", "life support",
        "critical", "hazardous", "high-risk", "catastrophic", "failure",
        "emergency", "abort", "escape", "redundancy",
    ],
    "human_in_loop": [
        "human-in-the-loop", "manual override", "operator", "pilot",
        "ground control", "decision", "judgment", "assessment",
    ],
}
