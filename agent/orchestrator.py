"""
orchestrator.py
---------------
Main workflow coordinator that passes data between the Risk Analyst,
Compliance, and Communication agents.
"""

from .risk_analyst_agent import run_risk_analyst_agent
from .compliance_agent import run_compliance_agent
from .communication_agent import run_communication_agent

def run_agent_pipeline(probability: float, shap_values: list, feature_names: list, business_rule: str, decision: str) -> dict:
    """
    Orchestrates the multi-agent workflow for credit decisioning.
    
    Parameters
    ----------
    probability : float
        The predicted probability of loan approval.
    shap_values : list
        The SHAP values for the specific applicant.
    feature_names : list
        The names of the features corresponding to the SHAP values.
    business_rule : str
        Any specific business rule that was triggered.
    decision : str
        The final system decision ("APPROVE", "REJECT", "REVIEW").
        
    Returns
    -------
    dict
        A dictionary containing the outputs of the various agents.
    """
    
    # 1. Risk Analyst summarizes the ML output
    risk_summary = run_risk_analyst_agent(
        probability=probability,
        shap_values=shap_values,
        feature_names=feature_names,
        business_rule=business_rule
    )
    
    # 2. Compliance Agent checks the summary for fair lending violations
    compliance_result = run_compliance_agent(risk_summary)
    
    status = compliance_result.get("status", "ERROR")
    notes = compliance_result.get("notes", "No notes provided.")
    
    final_letter = None
    
    # 3 & 4. Route based on compliance check
    if status == "APPROVED":
        # Safe to generate the applicant letter
        final_letter = run_communication_agent(
            decision=decision,
            risk_summary=risk_summary
        )
    elif status == "FLAGGED":
        # Do not generate letter, flag for human review
        final_letter = "LETTER GENERATION BLOCKED: Compliance flagged this decision for manual review."
    else:
        final_letter = f"LETTER GENERATION BLOCKED: Compliance check failed ({notes})."
        
    # 5. Return the full pipeline trace
    return {
        "risk_summary": risk_summary,
        "compliance_status": status,
        "compliance_notes": notes,
        "final_letter": final_letter
    }
