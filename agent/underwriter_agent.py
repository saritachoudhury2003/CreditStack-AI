"""
underwriter_agent.py
--------------------
Interactive agent that allows a human underwriter to ask follow-up questions,
look up documents, or simulate alternative loan scenarios.
"""

import ollama
import json
from . import tools

def run_underwriter_agent(user_question: str, applicant_context: dict) -> str:
    """
    Interactive agent for underwriters, capable of calling tools.
    
    Parameters
    ----------
    user_question : str
        The query from the human underwriter.
    applicant_context : dict
        Context including probability, shap_values, feature_names, and applicant_id, 
        plus any base features needed for simulation.
        
    Returns
    -------
    str
        The agent's response to the underwriter.
    """
    
    # 1. Define tools for Ollama
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "simulate_alternate_scenario",
                "description": "Simulates modifying an applicant's income and DTI to see how it affects their approval probability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_income": {
                            "type": "number",
                            "description": "The new total income to simulate."
                        },
                        "new_dti": {
                            "type": "number",
                            "description": "The new Debt-to-Income (Loan_Income_Ratio) to simulate."
                        }
                    },
                    "required": ["new_income", "new_dti"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_applicant_docs",
                "description": "Looks up the verification status of an applicant's submitted documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "applicant_id": {
                            "type": "string",
                            "description": "The ID of the applicant."
                        }
                    },
                    "required": ["applicant_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_shap_explanation",
                "description": "Gets the top 3 driving features for this applicant's risk score.",
                "parameters": {
                    "type": "object",
                    "properties": {}, # Takes data from context, no args needed from LLM
                }
            }
        }
    ]
    
    # Generate SHAP summary to include directly in prompt context
    shap_summary = tools.get_shap_explanation(
        applicant_context.get('applicant_id', 'Unknown'),
        applicant_context.get('shap_values', []),
        applicant_context.get('feature_names', [])
    )
    
    system_prompt = f"""
    You are an AI Underwriting Assistant. You help human underwriters evaluate loan applications.
    
    Applicant Context:
    - Applicant ID: {applicant_context.get('applicant_id', 'Unknown')}
    - Approval Probability: {applicant_context.get('probability', 0.0):.2%}
    - Risk Factor Analysis (SHAP):
    {shap_summary}
    
    Instructions:
    - If the user asks about risk factors, main drivers, or why the applicant was approved/rejected, explain the Risk Factor Analysis above clearly.
    - If the user asks about document verification status, call the fetch_applicant_docs tool.
    - If the user asks to simulate a new income or DTI scenario, call the simulate_alternate_scenario tool.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    try:
        # Initial call to see if a tool is needed
        response = ollama.chat(
            model="llama3.2:1b",
            messages=messages,
            tools=tool_schemas
        )
        
        # 3. Check if the model wants to call a tool
        if response.get("message", {}).get("tool_calls"):
            # Add the model's tool call request to the message history
            messages.append(response["message"])
            
            for tool_call in response["message"]["tool_calls"]:
                function_name = tool_call["function"]["name"]
                args = tool_call["function"].get("arguments", {})
                
                # Execute the specific tool
                result = ""
                if function_name == "simulate_alternate_scenario":
                    result = tools.simulate_alternate_scenario(
                        new_income=args.get("new_income"),
                        new_dti=args.get("new_dti"),
                        base_applicant_data=applicant_context.get("base_data", {})
                    )
                elif function_name == "fetch_applicant_docs":
                    result = tools.fetch_applicant_docs(args.get("applicant_id"))
                elif function_name == "get_shap_explanation":
                    result = tools.get_shap_explanation(
                        applicant_id=applicant_context.get("applicant_id"),
                        shap_values=applicant_context.get("shap_values", []),
                        feature_names=applicant_context.get("feature_names", [])
                    )
                else:
                    result = f"Unknown tool: {function_name}"
                
                # Add the tool result back to the conversation
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "name": function_name # Optional, but good practice
                })
            
            # Call Ollama again so it can formulate a final answer using the tool results
            final_response = ollama.chat(
                model="llama3.2:1b",
                messages=messages
            )
            return final_response["message"]["content"].strip()
            
        else:
            # 4. No tool call needed, just return the text
            return response["message"]["content"].strip()
            
    except Exception as e:
        # 5. Handle connection errors
        return f"Error connecting to Ollama. Ensure 'llama3.2:1b' is running locally. Details: {e}"
