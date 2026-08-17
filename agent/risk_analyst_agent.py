"""
risk_analyst_agent.py
---------------------
Agent responsible for translating technical ML outputs (probabilities and SHAP values)
into a human-readable risk summary.
"""

import ollama

def run_risk_analyst_agent(probability: float, shap_values: list, feature_names: list, business_rule: str) -> str:
    """
    Takes model outputs and uses Ollama to generate a plain-English risk summary.
    
    Parameters
    ----------
    probability : float
        The predicted probability of loan approval.
    shap_values : list
        The SHAP values for the specific applicant.
    feature_names : list
        The names of the features corresponding to the SHAP values.
    business_rule : str
        Any specific business rule that was triggered (e.g., "High DTI").
        
    Returns
    -------
    str
        A 2-4 sentence summary explaining the risk drivers.
    """
    # Build a simple representation of the top 3 contributing features
    # Combine feature names and their absolute SHAP impact to find top drivers
    impacts = list(zip(feature_names, shap_values))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3 = impacts[:3]
    
    feature_str = ", ".join([f"{name} (Impact: {val:.3f})" for name, val in top_3])

    prompt = f"""
    Applicant Data:
    - Predicted Approval Probability: {probability:.2%}
    - Top Risk Drivers (SHAP): {feature_str}
    - Business Rule Triggered: {business_rule}
    
    Task: Write a plain, simple English summary (2-4 sentences) explaining the main reasons for this applicant's risk profile based on the data above.
    """
    
    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert credit risk analyst. Your job is to explain complex machine learning risk drivers in simple, non-technical terms to underwriters."
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
