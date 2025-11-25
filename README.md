# tasker.ai

![Tasker.ai Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGNhYjQ0YjYxYzM5NjZkNTYyYjU3ODg2YjM4ZWMxNjQzZDRjYjBhZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L4Z4iSCoY2v5V6e4H2/giphy.gif)

## Overview
tasker.ai is an intelligent, AI-powered project management and task delegation platform. It uses NLP to automate project planning, task breakdown, and resource allocation.

## Features
*   **AI-Powered Task Generation:** Automatically generate a list of tasks from a project description.
*   **Intelligent Task Assignment:** Assign tasks to the most suitable employees based on their skills using semantic similarity.
*   **Dynamic Task Management:** Manually add new tasks to a project and have them intelligently assigned.
*   **Flexible Project Definition:** Define your own projects or choose from a list of default projects.
*   **Easy Data Input:** Upload employee data from a CSV file.
*   **Downloadable Reports:** Download the final task assignments as a CSV report.

## How to Run
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
3.  Open the app in your browser, enter your Google API key, upload an employee CSV, define a project, and generate tasks.

## Employee CSV Format
The employee CSV should have the following format:
```csv
name,skills
John Doe,"Python, Data Analysis, Machine Learning"
Jane Smith,"JavaScript, React, Web Development"
```
