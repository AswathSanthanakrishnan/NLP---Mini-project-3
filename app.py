
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="tasker.ai", layout="wide")

# ==============================================================================
# IMPORTANT: MANUAL STEP REQUIRED
# ==============================================================================
# To make the AI features work, you MUST replace the placeholder text below
# with your actual Hugging Face API key.
#
# 1. Get your API key from Hugging Face: https://huggingface.co/settings/tokens
# 2. Replace "YOUR_HUGGING_FACE_API_KEY_GOES_HERE" with your key.
#
# The app will NOT work until you do this.
# ==============================================================================
HF_API_KEY = "YOUR_HUGGING_FACE_API_KEY_GOES_HERE"


# --- Main App ---
def main():
    st.title("tasker.ai")
    st.write(
        "An intelligent, AI-powered project management and task delegation platform."
    )

    # --- Sidebar ---
    st.sidebar.title("Controls")
    st.sidebar.subheader("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload Employee CSV", type=["csv"])
    if uploaded_file:
        try:
            employees_df = pd.read_csv(uploaded_file)
            st.session_state["employees_df"] = employees_df
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

    # --- Main Content ---
    tab1, tab2, tab3 = st.tabs(
        ["Project & PRD", "Task Generation", "Employees & Assignments"]
    )

    # --- Project & PRD Tab ---
    with tab1:
        st.header("Project & PRD")
        default_projects = {
            "Select a project": "",
            "AI-Powered Customer Support Chatbot": "Develop a chatbot that can answer customer questions and resolve common issues. The chatbot should be able to understand natural language and provide personalized responses.",
            "E-commerce Website Redesign": "Redesign an e-commerce website to improve the user experience and increase sales. The project will involve updating the UI, improving navigation, and adding new features.",
            "Mobile App for Task Management": "Create a mobile app that helps users organize their tasks and stay productive. The app should have features like task creation, deadlines, and reminders.",
        }
        project_choice = st.selectbox(
            "Choose a default project", list(default_projects.keys())
        )
        project_name = st.text_input(
            "Project Name",
            value=project_choice if project_choice != "Select a project" else "",
        )
        project_description = st.text_area(
            "Project Description",
            value=default_projects[project_choice]
            if project_choice != "Select a project"
            else "",
            height=200,
        )

        if st.button("Generate PRD"):
            if not project_name or not project_description:
                st.error("Please enter a project name and description.")
            else:
                generate_prd(project_name, project_description)

        if "prd" in st.session_state:
            st.subheader("Generated PRD")
            st.markdown(st.session_state["prd"])

    # --- Task Generation Tab ---
    with tab2:
        st.header("Task Generation")
        prd_input = st.text_area(
            "Paste PRD here", value=st.session_state.get("prd", ""), height=400
        )
        if st.button("Generate Tasks"):
            if not prd_input:
                st.error("Please paste the PRD in the text area.")
            else:
                generate_tasks_from_prd(prd_input)

        if "tasks" in st.session_state:
            st.subheader("Generated Tasks")
            st.write(st.session_state["tasks"])

    # --- Employees & Assignments Tab ---
    with tab3:
        st.header("Employees & Assignments")
        if "employees_df" in st.session_state:
            st.subheader("Available Employees")
            st.dataframe(st.session_state["employees_df"], use_container_width=True)

            if "tasks" in st.session_state:
                if st.button("Assign Tasks"):
                    assign_tasks(
                        st.session_state["tasks"], st.session_state["employees_df"]
                    )

            if "assignments_df" in st.session_state:
                st.subheader("Task Assignments")
                st.dataframe(
                    st.session_state["assignments_df"], use_container_width=True
                )
                csv = st.session_state["assignments_df"].to_csv(index=False)
                st.download_button(
                    "Download Assignments as CSV",
                    csv,
                    "task_assignments.csv",
                    "text/csv",
                )
        else:
            st.warning("Please upload an employee CSV file in the sidebar.")


def generate_from_huggingface(prompt):
    if not HF_API_KEY or HF_API_KEY == "YOUR_HUGGING_FACE_API_KEY_GOES_HERE":
        st.error(
            "Please replace 'YOUR_HUGGING_FACE_API_KEY_GOES_HERE' in the app.py file with your actual Hugging Face API key."
        )
        return None

    try:
        client = InferenceClient(token=HF_API_KEY)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating text from Hugging Face: {e}")
        return None


def generate_prd(project_name, project_description):
    st.info("Generating PRD...")
    prompt = f"Generate a Product Requirements Document (PRD) for a project named '{project_name}' with the description: '{project_description}'. The PRD should include sections for Overview, Core Features, and Technical Architecture."
    prd = generate_from_huggingface(prompt)
    if prd:
        st.session_state["prd"] = prd

def generate_tasks_from_prd(prd_input):
    st.info("Generating tasks...")
    prompt = f"Generate a list of tasks from the following PRD. Provide the tasks as a numbered list.\n\n{prd_input}"
    tasks_text = generate_from_huggingface(prompt)
    if tasks_text:
        tasks = tasks_text.strip().split("\n")
        st.session_state["tasks"] = tasks

def assign_tasks(tasks, employees_df):
    st.info("Assigning tasks...")
    assignments = []
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        employee_skills = employees_df["skills"].tolist()
        skill_embeddings = model.encode(employee_skills, convert_to_tensor=True)

        for task in tasks:
            task_embedding = model.encode(task, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(task_embedding, skill_embeddings)
            best_employee_idx = np.argmax(cosine_scores)
            assignments.append(
                {
                    "Task": task,
                    "Assigned To": employees_df.iloc[best_employee_idx]["name"],
                    "Confidence": f"{cosine_scores[0][best_employee_idx]:.2f}",
                }
            )
        assignments_df = pd.DataFrame(assignments)
        st.session_state["assignments_df"] = assignments_df
    except Exception as e:
        st.error(f"Error assigning tasks: {e}")


if __name__ == "__main__":
    main()
