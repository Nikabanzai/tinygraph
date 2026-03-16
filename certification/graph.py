"""Certification subgraph definition.

Implements PRC (Project Classification) and CEP (Certification Programme)
for aerospace project certification.

PRC: Classifies projects as Minor or Major based on:
     - Novel technology involvement
     - High system complexity
     - Critical safety impacts
     - Need for human-in-the-loop logic

CEP (Certification Programme): For Major projects, routes through:
     - cert_aerospace: Aerospace compliance checks
     - cert_mission: Mission requirements validation
     - cert_safety: Safety & risk assessment
"""

from __future__ import annotations

import json
from typing import TypedDict, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from certification.state import CertificationState, CERTIFICATION_STAGES, MAJOR_CRITERIA_KEYWORDS


class CertificationContext(TypedDict):
    """Runtime context passed from the parent graph."""

    model: object


def extract_json(text: str) -> dict:
    """Extract JSON payload from an LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def classify_project_prc(project_description: str, model) -> tuple[str, str]:
    """Classify project as Minor or Major using LLM (PRC - Project Classification).

    A project is "Major" if it involves:
    - Novel technology (unproven, experimental, first-of-its-kind)
    - High system complexity (multiple systems, integrated, sophisticated)
    - Critical safety impacts (human-rated, crew, life support, hazardous)
    - Human-in-the-loop logic essential for certification

    Returns:
        Tuple of (classification, reasoning)
    """
    if model is None:
        # Default to Major if model unavailable (safer default)
        return "Major", "Model unavailable - defaulting to Major for safety"

    # Build criteria hints for the LLM
    criteria_hints = []
    desc_lower = project_description.lower()

    for category, keywords in MAJOR_CRITERIA_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in desc_lower]
        if matched:
            criteria_hints.append(f"- {category.replace('_', ' ').title()}: {', '.join(matched[:3])}")

    criteria_text = "\n".join(criteria_hints) if criteria_hints else "No specific criteria keywords detected."

    prompt = f"""You are a Project Classification (PRC) system for aerospace certification.

Analyze the following project description and classify it as either "Minor" or "Major".

Project Description:
{project_description}

Detected Criteria Hints:
{criteria_text}

Classification Criteria:
A project is "Major" if ANY of the following apply:
1. Novel Technology: Involves unproven, experimental, or first-of-its-kind technology
2. High Complexity: Multiple integrated systems, sophisticated, or multi-stage
3. Critical Safety: Human-rated, crew/passenger carrying, life support, or hazardous operations
4. Human-in-the-Loop: Requires human judgment, decision-making, or oversight for certification

If NONE of these apply significantly, classify as "Minor".

Respond with ONLY valid JSON:
{{"classification": "Minor" or "Major", "reasoning": "<brief explanation of why>"}}
"""

    response = model(prompt, max_tokens=200, temperature=0.0, stop=["\n\n"])
    content = response["choices"][0]["text"].strip()
    data = extract_json(content)

    classification = data.get("classification", "Major")  # Default to Major for safety
    reasoning = data.get("reasoning", "Classification based on project analysis")

    # Validate classification
    if classification not in ["Minor", "Major"]:
        classification = "Major"  # Default to Major for safety

    return classification, reasoning


def prc_triage_node(state: CertificationState, *, runtime) -> CertificationState:
    """PRC (Project Classification) triage node.

    Uses LLM to classify the project as Minor or Major.
    Minor projects get a simple report.
    Major projects proceed to CEP (Certification Programme).
    """
    project_description = state.get("project_description", state.get("user_query", ""))
    model = runtime.context["model"] if runtime.context else None

    # Classify the project using PRC
    classification, reasoning = classify_project_prc(project_description, model)

    messages = state.get("messages", [])
    classification_msg = AIMessage(
        content=f"[PRC] Project Classification: {classification}\n\nReasoning: {reasoning}"
    )
    messages = messages + [classification_msg]

    if classification == "Major":
        # Route to CEP (Certification Programme)
        return {
            "messages": messages,
            "project_description": project_description,
            "project_classification": "Major",
            "classification_reasoning": reasoning,
            "certification_status": "Pending",
            "certification_stage": "CEP - Starting Aerospace Compliance",
            "next_step": "cert_aerospace",
            "tool_attempts": 0,
            "tool_failed": False,
            "tool_error": "",
        }
    else:
        # Minor project - simple certification
        minor_report = f"""[CERTIFICATION REPORT - MINOR PROJECT]

Project: {project_description[:100]}...

Classification: Minor
Reasoning: {reasoning}

Status: PASSED (Minor Project - Standard Compliance)
- No novel technology detected
- Standard complexity level
- No critical safety impacts requiring human-in-the-loop

Recommendation: Proceed with standard certification process.
No full CEP (Certification Programme) required.
"""
        return {
            "messages": messages + [AIMessage(content=minor_report)],
            "project_description": project_description,
            "project_classification": "Minor",
            "classification_reasoning": reasoning,
            "certification_status": "Passed",
            "certification_stage": "Complete",
            "certification_report": minor_report,
            "next_step": "end",
            "tool_attempts": 0,
            "tool_failed": False,
            "tool_error": "",
        }


def cert_aerospace_node(state: CertificationState, *, runtime) -> CertificationState:
    """CEP: Aerospace Compliance node.

    Validates project against aerospace standards and regulations.
    """
    project_description = state.get("project_description", "")
    model = runtime.context["model"] if runtime.context else None

    if model is None:
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content="[CERT_AEROSPACE] Model unavailable - skipping aerospace compliance check")
            ],
            "certification_stage": "CEP - Aerospace Compliance (Skipped)",
            "next_step": "cert_mission",
            "aerospace_compliance": {"status": "skipped", "reason": "model_unavailable"},
        }

    prompt = f"""You are an Aerospace Compliance Certification system (CEP - Aerospace).

Analyze the following project for compliance with aerospace standards:

Project Description:
{project_description}

Check for:
1. Materials and components standards compliance
2. Environmental qualification requirements
3. Electromagnetic compatibility
4. Structural integrity requirements
5. Thermal and pressure requirements

Respond with ONLY valid JSON:
{{"compliance_status": "Passed" or "Failed" or "Conditional", "findings": "<brief findings>", "requirements": ["req1", "req2"]}}
"""

    response = model(prompt, max_tokens=200, temperature=0.0, stop=["\n\n"])
    content = response["choices"][0]["text"].strip()
    data = extract_json(content)

    compliance_status = data.get("compliance_status", "Conditional")
    findings = data.get("findings", "Aerospace compliance assessment completed")
    requirements = data.get("requirements", [])

    messages = state.get("messages", [])
    aerospace_msg = AIMessage(
        content=f"[CEP] Aerospace Compliance: {compliance_status}\n\nFindings: {findings}"
    )

    compliance_result = {
        "status": compliance_status,
        "findings": findings,
        "requirements": requirements,
    }

    # Determine next step based on compliance
    if compliance_status == "Failed":
        return {
            "messages": messages + [aerospace_msg],
            "certification_status": "Failed",
            "certification_stage": "CEP - Aerospace Compliance (Failed)",
            "aerospace_compliance": compliance_result,
            "certification_report": f"Aerospace compliance FAILED: {findings}",
            "next_step": "end",
        }

    return {
        "messages": messages + [aerospace_msg],
        "certification_stage": "CEP - Aerospace Compliance (Passed)",
        "aerospace_compliance": compliance_result,
        "next_step": "cert_mission",
    }


def cert_mission_node(state: CertificationState, *, runtime) -> CertificationState:
    """CEP: Mission Requirements node.

    Validates project against mission-specific requirements.
    """
    project_description = state.get("project_description", "")
    model = runtime.context["model"] if runtime.context else None

    if model is None:
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content="[CERT_MISSION] Model unavailable - skipping mission requirements check")
            ],
            "certification_stage": "CEP - Mission Requirements (Skipped)",
            "next_step": "cert_safety",
            "mission_requirements": {"status": "skipped", "reason": "model_unavailable"},
        }

    prompt = f"""You are a Mission Requirements Certification system (CEP - Mission).

Analyze the following project for mission requirements compliance:

Project Description:
{project_description}

Check for:
1. Mission objectives alignment
2. Performance requirements
3. Operational constraints
4. Interface requirements
5. Timeline and schedule feasibility

Respond with ONLY valid JSON:
{{"mission_status": "Passed" or "Failed" or "Conditional", "findings": "<brief findings>", "critical_requirements": ["req1", "req2"]}}
"""

    response = model(prompt, max_tokens=200, temperature=0.0, stop=["\n\n"])
    content = response["choices"][0]["text"].strip()
    data = extract_json(content)

    mission_status = data.get("mission_status", "Conditional")
    findings = data.get("findings", "Mission requirements assessment completed")
    critical_requirements = data.get("critical_requirements", [])

    messages = state.get("messages", [])
    mission_msg = AIMessage(
        content=f"[CEP] Mission Requirements: {mission_status}\n\nFindings: {findings}"
    )

    mission_result = {
        "status": mission_status,
        "findings": findings,
        "critical_requirements": critical_requirements,
    }

    if mission_status == "Failed":
        return {
            "messages": messages + [mission_msg],
            "certification_status": "Failed",
            "certification_stage": "CEP - Mission Requirements (Failed)",
            "mission_requirements": mission_result,
            "certification_report": f"Mission requirements FAILED: {findings}",
            "next_step": "end",
        }

    return {
        "messages": messages + [mission_msg],
        "certification_stage": "CEP - Mission Requirements (Passed)",
        "mission_requirements": mission_result,
        "next_step": "cert_safety",
    }


def cert_safety_node(state: CertificationState, *, runtime) -> CertificationState:
    """CEP: Safety Assessment node.

    Performs safety and risk assessment for the project.
    """
    project_description = state.get("project_description", "")
    model = runtime.context["model"] if runtime.context else None

    if model is None:
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content="[CERT_SAFETY] Model unavailable - skipping safety assessment")
            ],
            "certification_stage": "CEP - Safety Assessment (Skipped)",
            "next_step": "cert_final",
            "safety_assessment": {"status": "skipped", "reason": "model_unavailable"},
        }

    prompt = f"""You are a Safety Assessment Certification system (CEP - Safety).

Analyze the following project for safety and risk assessment:

Project Description:
{project_description}

Check for:
1. Hazard identification and mitigation
2. Risk assessment (probability x severity)
3. Safety margins and redundancy
4. Failure mode analysis
5. Emergency procedures adequacy

Respond with ONLY valid JSON:
{{"safety_status": "Passed" or "Failed" or "Conditional", "risk_level": "Low" or "Medium" or "High" or "Critical", "findings": "<brief findings>", "recommendations": ["rec1", "rec2"]}}
"""

    response = model(prompt, max_tokens=250, temperature=0.0, stop=["\n\n"])
    content = response["choices"][0]["text"].strip()
    data = extract_json(content)

    safety_status = data.get("safety_status", "Conditional")
    risk_level = data.get("risk_level", "Medium")
    findings = data.get("findings", "Safety assessment completed")
    recommendations = data.get("recommendations", [])

    messages = state.get("messages", [])
    safety_msg = AIMessage(
        content=f"[CEP] Safety Assessment: {safety_status}\nRisk Level: {risk_level}\nFindings: {findings}"
    )

    safety_result = {
        "status": safety_status,
        "risk_level": risk_level,
        "findings": findings,
        "recommendations": recommendations,
    }

    if safety_status == "Failed":
        return {
            "messages": messages + [safety_msg],
            "certification_status": "Failed",
            "certification_stage": "CEP - Safety Assessment (Failed)",
            "safety_assessment": safety_result,
            "certification_report": f"Safety assessment FAILED: {findings}",
            "next_step": "end",
        }

    return {
        "messages": messages + [safety_msg],
        "certification_stage": "CEP - Safety Assessment (Passed)",
        "safety_assessment": safety_result,
        "next_step": "cert_final",
    }


def human_review_node(state: CertificationState, *, runtime) -> CertificationState:
    """HITL Governance node for MAJOR project certification.

    Task 2: Implement HITL Governance for Certification
    This node pauses the graph for human review.
    The actual approval/rejection is handled by the TUI modal.
    """
    project_description = state.get("project_description", "")
    classification_reasoning = state.get("classification_reasoning", "")

    messages = state.get("messages", [])
    review_msg = AIMessage(
        content=f"[HUMAN REVIEW REQUIRED]\n\n"
                f"Project: {project_description[:100]}...\n"
                f"Classification: MAJOR\n"
                f"Reasoning: {classification_reasoning}\n\n"
                f"Awaiting operator certification decision..."
    )

    return {
        "messages": messages + [review_msg],
        "certification_stage": "Human Review - Pending",
        "pending_action": f"Certify MAJOR project: {project_description[:50]}...",
        "next_step": "human_review",
    }


def cert_final_node(state: CertificationState, *, runtime) -> CertificationState:
    """CEP: Final Certification node.

    Generates the final certification report.
    """
    project_description = state.get("project_description", "")
    aerospace = state.get("aerospace_compliance", {})
    mission = state.get("mission_requirements", {})
    safety = state.get("safety_assessment", {})
    operator_approved = state.get("operator_approved")

    # Check if operator rejected
    if operator_approved is False:
        status = "Rejected"
        report = f"""[CERTIFICATION REJECTED]

Project: {project_description[:100]}...
Classification: MAJOR
Status: REJECTED BY OPERATOR

The human reviewer has rejected certification for this project.
"""
        messages = state.get("messages", [])
        final_msg = AIMessage(content=report)
        return {
            "messages": messages + [final_msg],
            "certification_status": "Rejected",
            "certification_stage": "CEP - Rejected",
            "certification_report": report,
            "next_step": "end",
        }

    # Determine overall status
    all_passed = all([
        aerospace.get("status") in ["Passed", "Conditional", None],
        mission.get("status") in ["Passed", "Conditional", None],
        safety.get("status") in ["Passed", "Conditional", None],
    ])

    if all_passed:
        status = "Passed"
    elif any([
        aerospace.get("status") == "Failed",
        mission.get("status") == "Failed",
        safety.get("status") == "Failed",
    ]):
        status = "Failed"
    else:
        status = "Pending"

    report = f"""╔══════════════════════════════════════════════════════════════════╗
║           CERTIFICATION PROGRAMME (CEP) - FINAL REPORT            ║
╠══════════════════════════════════════════════════════════════════╣
║ Project: {project_description[:45]:<45} ║
║ Classification: Major (Full CEP Required)                        ║
║ Operator Approved: {"YES" if operator_approved else "PENDING":<40} ║
╠══════════════════════════════════════════════════════════════════╣
║ AEROSPACE COMPLIANCE: {aerospace.get('status', 'N/A'):<40} ║
║ MISSION REQUIREMENTS: {mission.get('status', 'N/A'):<40} ║
║ SAFETY ASSESSMENT:    {safety.get('status', 'N/A'):<40} ║
╠══════════════════════════════════════════════════════════════════╣
║ OVERALL STATUS: {status:<50} ║
╚══════════════════════════════════════════════════════════════════╝

Findings Summary:
- Aerospace: {aerospace.get('findings', 'N/A')}
- Mission: {mission.get('findings', 'N/A')}
- Safety: {safety.get('findings', 'N/A')}

Certification Programme Complete.
"""

    messages = state.get("messages", [])
    final_msg = AIMessage(content=report)

    return {
        "messages": messages + [final_msg],
        "certification_status": status,
        "certification_stage": "CEP - Complete",
        "certification_report": report,
        "next_step": "end",
    }


def build_certification_graph(checkpointer: Checkpointer | None = None):
    """Build and compile the certification subgraph.

    Task 2: Implement HITL Governance for Certification
    Flow:
    START -> prc_triage -> (Minor: end) or (Major: human_review -> cert_aerospace -> ...) -> END
    Uses interrupt_before=["human_review"] for HITL pause.
    """
    builder = StateGraph(CertificationState, context_schema=CertificationContext)

    # Add nodes
    builder.add_node("prc_triage", prc_triage_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("cert_aerospace", cert_aerospace_node)
    builder.add_node("cert_mission", cert_mission_node)
    builder.add_node("cert_safety", cert_safety_node)
    builder.add_node("cert_final", cert_final_node)

    # Add edges
    builder.add_edge(START, "prc_triage")

    # Conditional routing from PRC triage
    # Task 2: MAJOR -> human_review (pauses), MINOR -> end
    def prc_router(state: CertificationState) -> Literal["human_review", "cert_final", "end"]:
        next_step = state.get("next_step", "end")
        classification = state.get("project_classification", "")

        if next_step == "human_review" or classification == "Major":
            return "human_review"
        elif next_step == "end":
            return "end"
        return "cert_final"

    builder.add_conditional_edges(
        "prc_triage",
        prc_router,
        {
            "human_review": "human_review",
            "cert_final": "cert_final",
            "end": END,
        },
    )

    # Human review routes to CEP or end based on operator decision
    def human_review_router(state: CertificationState) -> Literal["cert_aerospace", "end"]:
        operator_approved = state.get("operator_approved")
        if operator_approved is False:
            return "end"
        return "cert_aerospace"

    builder.add_conditional_edges(
        "human_review",
        human_review_router,
        {
            "cert_aerospace": "cert_aerospace",
            "end": END,
        },
    )

    # CEP flow for Major projects
    builder.add_edge("cert_aerospace", "cert_mission")
    builder.add_edge("cert_mission", "cert_safety")
    builder.add_edge("cert_safety", "cert_final")
    builder.add_edge("cert_final", END)

    # Task 2: Interrupt before human_review for HITL governance
    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


certification_graph = build_certification_graph()
