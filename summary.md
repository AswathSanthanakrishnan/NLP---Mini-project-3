tasker.ai Project Report

1. Introduction  
tasker.ai is a prototype that explores whether lightweight NLP models can help with early-stage project planning. The goal is to automate three steps that are usually manual: drafting a product requirements document (PRD), turning that PRD into a structured task list, and assigning those tasks to team members based on their skills. The work was done as a course project to practice applied NLP and rapid prototyping with minimal infrastructure.

2. System Overview  
The application is a single Streamlit app (`app.py`). It runs locally and keeps all state in Streamlit session state. The workflow is split into three tabs: (a) Project and PRD creation, (b) Task generation, and (c) Employee uploads plus task assignment. Users upload a CSV of employees with names and skill strings, describe a project, generate a PRD, generate tasks, and then assign tasks.

3. Models and Libraries  
- Text generation: `distilgpt2` via Hugging Face `transformers` and `accelerate`, used to expand a project description into PRD sections and to draft tasks from that PRD.  
- Task-to-employee matching: `all-MiniLM-L6-v2` from `sentence-transformers`, using cosine similarity (`util.pytorch_cos_sim`) between task embeddings and skill embeddings.  
- Frameworks: Streamlit for the UI, pandas for CSV handling, NumPy for array ops, and PyTorch as the backend for both models.

4. Implementation Details  
- PRD generation: Prompts are constructed from the project name and description. The generated text is post-processed to pull bullet points for features and tools, with keyword-based fallbacks for common domains (e.g., mobile, web, chatbot).  
- Task generation: Tasks are derived both from the PRD text (parsing feature and tool sections) and from model output seeded by the PRD. Duplicate removal keeps the list concise.  
- Task assignment: Employee skills are embedded once, then each task is embedded and matched to the highest cosine similarity. The UI shows assigned tasks, matched skills, and a confidence score, and provides a CSV download.  
- Caching: Streamlit caching is used for the text-generation model; other state lives in `st.session_state`.

5. Usage  
Prerequisites: Python 3.9+ with dependencies pinned in `requirements.txt` (Streamlit, pandas, transformers, accelerate, torch, sentence-transformers). On first run, the two models download from Hugging Face (~hundreds of MB).  
Run: `streamlit run app.py`, then upload `name,skills` CSV, enter a project name/description, generate the PRD, generate tasks, and assign them.

6. Results and Observations  
- The prototype reliably produces a coherent PRD and a task list for typical software projects (mobile, web, chatbot).  
- Matching quality depends heavily on the specificity of the `skills` column; detailed skills improve alignment.  
- Distilgpt2 is small and runs locally, but long prompts can lead to verbose outputs; limiting max tokens keeps latency acceptable.  
- The end-to-end flow demonstrates the feasibility of pairing generative text with semantic matching for lightweight project planning.

7. Limitations  
- No persistence beyond the current session; refreshing the page resets state.  
- Model outputs can still be verbose or off-topic; there is no strong guardrail or schema enforcement.  
- Employee matching assumes a single “best” assignee per task and does not consider availability, workload, or multiple assignees.  
- Cold start requires downloading models, which may be slow on constrained networks.

8. Future Work  
- Add schema-aware task generation (e.g., ask for JSON and validate) to reduce noise.  
- Cache sentence-transformer embeddings and add multi-employee suggestions with tie-breaking by secondary skills.  
- Support workload and availability constraints in assignment.  
- Add persistence (e.g., SQLite) and auth to retain projects across sessions.  
- Offer a smaller or quantized model option for slower machines, and a toggle for offline use if models are pre-fetched.

9. Conclusion  
tasker.ai shows that small, local NLP models can automate a slice of project planning: turning descriptions into structured artifacts and matching them to people. While the current prototype is intentionally simple, it provides a base to explore more rigorous task schemas, richer matching signals, and production readiness.
