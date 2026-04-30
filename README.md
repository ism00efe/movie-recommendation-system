# RepoReader AI

## 1. Project Overview
RepoReader AI is a repository analysis system that fetches selected files from public GitHub repositories, processes them with Gemini 2.5 Flash, and returns structured architecture and security insights.  
It extracts key project files through the GitHub Contents API, performs autonomous code analysis, and generates a Mermaid-compatible architecture diagram alongside security alerts.  
This project was built for the MLH AI Hackfest.

## 2. System Architecture & Flow
- **Frontend (Streamlit):** Accepts a GitHub repository URL from the user and sends it to the backend.
- **Backend (FastAPI):** Parses `owner/repo` from the URL, requests target files from GitHub, and Base64-decodes file content.
- **AI Agent (Gemini 2.5 Flash):** Analyzes aggregated repository context and returns strict JSON with `summary`, `security_alerts`, and `mermaid_diagram`.
- **UI Rendering:** Mermaid code is sanitized and rendered inside an HTML widget in Streamlit.

## 3. Tech Stack
- Python
- FastAPI
- Streamlit
- Google Generative AI (`google-generativeai`)
- Mermaid.js (rendered via Streamlit HTML component)
- Requests
- Pydantic
- Python Dotenv

## 4. Installation & Setup
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd repo-reader-ai
   ```
2. Create a `.env` file in the project root and set:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```
   `GEMINI_API_KEY` is mandatory. The application raises an error at startup if it is missing.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 5. Running the Application
Run backend and frontend in separate terminals.

- **Terminal 1 (Backend):**
  ```bash
  uvicorn main:app --reload --port 8000
  ```
- **Terminal 2 (Frontend):**
  ```bash
  streamlit run frontend.py
  ```

The frontend sends requests to:
`http://127.0.0.1:8000/api/analyze`

## 6. LLM Prompt Strategy
The AI agent is constrained with deterministic prompt rules to control output quality and parsing reliability:
- Repository files are concatenated into a single context block with per-file boundaries.
- Token optimization is applied by truncating each file to **10KB** (`10000` characters) when needed.
- The model is forced to return JSON (`response_mime_type = application/json`) and parsed as structured data.
- A strict schema is enforced with exactly three outputs:
  - `summary` (2-3 sentences)
  - `security_alerts` (list of findings)
  - `mermaid_diagram` (simple `graph TD` syntax)
- Mermaid generation is restricted with hard rules to reduce rendering failures:
  - no brackets `[]`
  - no parentheses `()`
  - alphabetical node names only
  - no spaces in node names
  - semicolon-separated statements

## 7. Future Improvements
- [ ] Add OAuth-based GitHub integration to support analysis of private repositories.
- [ ] Add RAG (Retrieval-Augmented Generation) for large repositories or migrate to models with larger token windows.
