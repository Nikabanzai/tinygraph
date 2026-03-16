"""Telemetry tools for mission control agent."""

from langchain_core.tools import tool


@tool
def read_pressure(id: str) -> str:
    """Read pressure sensor by identifier."""
    return "340 PSI - Stable"


@tool
def check_fuel() -> str:
    """Check fuel levels for main and RCS tanks."""
    return "Main: 67%, RCS: 88%"


@tool
def log_anomaly(details: str) -> str:
    """Report anomaly details to the ground station."""
    return "Anomaly reported to ground station."
