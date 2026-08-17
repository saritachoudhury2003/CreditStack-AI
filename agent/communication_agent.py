"""
communication_agent.py
----------------------
Agent responsible for drafting professional letters to applicants.
"""

import ollama

def run_communication_agent(decision: str, risk_summary: str) -> str:
    """
    Drafts an applicant-facing letter based on the final decision and risk summary.
    
    Parameters
    ----------
    decision : str
        The final decision: "APPROVE", "REJECT", or "REVIEW".
    risk_summary : str
        The compliance-approved summary of risk drivers.
        
    Returns
    -------
    str
        A professional letter addressed to the applicant.
    """
    prompt = f"""
    Decision: {decision}
    Internal Risk Summary: {risk_summary}
    
    Task: Draft a short, polite, professional letter to the applicant informing them of the decision. 
    If approved, congratulate them.
    If rejected, gently explain the reasons based strictly on the Internal Risk Summary.
    If under review, ask them to wait for further contact.
    Keep it under 3 paragraphs.
    """
    
    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional customer communications representative for a bank. You write clear, empathetic, and professional letters."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Error connecting to Ollama. Please ensure Ollama is running locally with the 'llama3.2:1b' model. Details: {e}"
