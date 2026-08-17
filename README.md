# CreditStack AI ⚡

CreditStack AI is a modern, end-to-end intelligent credit risk assessment platform. It combines traditional Machine Learning pipelines (XGBoost, Stacking Ensembles) with Explainable AI (SHAP) and a local Multi-Agent LLM Orchestrator to provide a comprehensive, transparent, and fair loan underwriting experience.

## ✨ Features

*   **Machine Learning Credit Scoring:** Compares baseline ML models with advanced stacking ensembles.
*   **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to break down exactly which applicant features drove the approval or rejection decision.
*   **AI Agent Swarm Orchestration:** Runs a live, real-time pipeline of specialized AI agents:
    *   🕵️ **Risk Analyst:** Translates raw SHAP values and probabilities into plain-English risk summaries.
    *   ⚖️ **Compliance Officer:** Audits the risk summary for fair-lending violations (e.g., protected demographic classes).
    *   ✉️ **Communication Agent:** Automatically drafts a professional applicant-facing letter if compliance passes.
*   **Custom Dataset Support:** Upload custom CSV datasets to instantly retrain the ML pipelines and evaluate candidates dynamically.
*   **Persistent Storage:** Uses SQLite to store applicant decisions, historical agent runs, and chat histories.

## 🛠️ Tech Stack

**Frontend:**
*   **Framework:** Next.js (App Router) & React
*   **Styling:** Tailwind CSS (Glassmorphism design system)
*   **Build Tool:** Turbopack

**Backend:**
*   **Framework:** FastAPI (Python)
*   **Database:** SQLite
*   **Machine Learning:** Scikit-Learn, XGBoost, Pandas, Numpy
*   **Explainability:** SHAP
*   **LLM Integration:** Ollama (running `llama3.2:1b` locally)
*   **Streaming:** Server-Sent Events (SSE) for real-time agent output streaming.

## 🚀 Getting Started

### Prerequisites
*   Node.js and npm
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed locally and running the `llama3.2:1b` model.

### 1. Start the Local LLM Engine
Ensure Ollama is installed and running in the background. Pull the required model:
```bash
ollama pull llama3.2:1b
```

### 2. Setup the Backend (FastAPI)
Navigate to the root directory and install Python dependencies:
```bash
python -m venv venv
# Activate the virtual environment:
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

Run the backend server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Setup the Frontend (Next.js)
Open a new terminal, navigate to the frontend directory, and install dependencies:
```bash
cd frontend_next
npm install
```

Start the development server:
```bash
npm run dev
```

### 4. Open the App
Visit [http://localhost:3000](http://localhost:3000) in your browser.

## 🗄️ Project Structure

*   `/api`: FastAPI backend routes and application logic.
*   `/agent`: Multi-agent definitions (Risk Analyst, Compliance, Communication, Underwriter).
*   `/db`: SQLite database initialization and queries.
*   `/frontend_next`: Next.js frontend application.
*   `/models`: Saved `.pkl` model files for ML pipelines.
*   `/processed_data`: Scalers and encoders from preprocessing.

## 📜 License
MIT License
