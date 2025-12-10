# Tasker.ai

![Tasker.ai Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGNhYjQ0YjYxYzM5NjZkNTYyYjU3ODg2YjM4ZWMxNjQzZDRjYjBhZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L4Z4iSCoY2v5V6e4H2/giphy.gif)

## Overview
Tasker.ai is an intelligent, AI-powered project management and task delegation platform. It uses NLP to automate project planning, task breakdown, and resource allocation.

## Features
*   **AI-Powered Task Generation:** Automatically generate a list of tasks from a project description.
*   **Intelligent Task Assignment:** Assign tasks to the most suitable employees based on their skills using semantic similarity.
*   **Dynamic Task Management:** Manually add new tasks to a project and have them intelligently assigned.
*   **Flexible Project Definition:** Define your own projects or choose from a list of default projects.
*   **Easy Data Input:** Upload employee data from a CSV file.
*   **Downloadable Reports:** Download the final task assignments as a CSV report.

## How to Run
1.  **Install dependencies (Python 3.9+):**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
3.  Open the app in your browser, upload an employee CSV, pick or define a project, generate a PRD, then generate and assign tasks.

### Model downloads
- Text generation uses the local Hugging Face model `distilgpt2`.
- Task-to-employee matching uses `all-MiniLM-L6-v2` from SentenceTransformers.
- On first run, these models download automatically to the Hugging Face cache (~hundreds of MB). Subsequent runs are local/offline as long as the cache persists.

## Employee CSV Format
The employee CSV should have the following format:
```csv
name,skills
John Doe,"Python, Data Analysis, Machine Learning"
Jane Smith,"JavaScript, React, Web Development"
```
