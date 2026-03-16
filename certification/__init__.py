"""Certification subgraph for aerospace project classification and certification.

PRC (Project Classification): Classifies projects as Minor or Major
CEP (Certification Programme): Full certification for Major projects

Modules:
- state: CertificationState TypedDict and constants
- graph: Certification graph with PRC triage and CEP nodes
"""

from certification.state import CertificationState, CERTIFICATION_STAGES, MAJOR_CRITERIA_KEYWORDS
from certification.graph import build_certification_graph, certification_graph

__all__ = [
    "CertificationState",
    "CERTIFICATION_STAGES",
    "MAJOR_CRITERIA_KEYWORDS",
    "build_certification_graph",
    "certification_graph",
]
