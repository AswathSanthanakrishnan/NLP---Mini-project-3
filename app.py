
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

# --- Configuration ---
st.set_page_config(page_title="tasker.ai", layout="wide")

# Initialize model cache in session state
if "text_generator" not in st.session_state:
    st.session_state["text_generator"] = None
    st.session_state["model_loaded"] = False


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


@st.cache_resource
def load_text_generator():
    """Load the text generation model (cached to avoid reloading)"""
    try:
        # Using distilgpt2 - smaller and faster than gpt2, good for basic text generation
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1 if not torch.cuda.is_available() else 0,  # Use CPU if no GPU
        )
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_from_model(prompt):
    """Generate text using local model"""
    try:
        # Load model if not already loaded
        if st.session_state["text_generator"] is None:
            with st.spinner("Loading AI model (first time only, this may take a moment)..."):
                generator = load_text_generator()
                if generator is None:
                    return None
                st.session_state["text_generator"] = generator
                st.session_state["model_loaded"] = True
        
        generator = st.session_state["text_generator"]
        
        # Format prompt for better generation
        formatted_prompt = prompt + "\n\n"
        
        # Generate text
        results = generator(
            formatted_prompt,
            max_length=len(formatted_prompt.split()) + 300,  # Generate additional tokens
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        
        generated_text = results[0]["generated_text"]
        
        # Remove the original prompt from the generated text
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        return generated_text
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return None


def generate_prd(project_name, project_description):
    st.info("Generating PRD...")
    # Create a structured PRD based on the project description
    # Since GPT-2 is not instruction-tuned, we'll create a template-based PRD
    # that incorporates the project description intelligently
    
    # Generate some features based on the description
    prompt = f"Features for {project_name}: {project_description[:100]}"
    features_text = generate_from_model(prompt)
    
    # Create a well-structured PRD
    features_list = []
    if features_text:
        # Extract potential features from generated text
        sentences = features_text.split('.')
        for sent in sentences[:5]:  # Take first 5 sentences as features
            sent = sent.strip()
            if len(sent) > 10:
                features_list.append(f"- {sent}")
    
    # If no features generated, create default ones based on keywords
    if not features_list:
        keywords = project_description.lower()
        if 'chatbot' in keywords or 'ai' in keywords:
            features_list = [
                "- Natural language processing capabilities",
                "- User-friendly conversational interface",
                "- Integration with knowledge base",
                "- Multi-channel support"
            ]
        elif 'e-commerce' in keywords or 'website' in keywords:
            features_list = [
                "- Responsive design for all devices",
                "- Secure payment processing",
                "- Product catalog management",
                "- Shopping cart functionality"
            ]
        elif 'mobile' in keywords or 'app' in keywords:
            features_list = [
                "- Cross-platform compatibility",
                "- Offline functionality",
                "- Push notifications",
                "- User authentication"
            ]
        else:
            features_list = [
                "- User-friendly interface",
                "- Core functionality implementation",
                "- Data management system",
                "- Security and authentication"
            ]
    
    full_prd = f"""# Product Requirements Document: {project_name}

## Overview
{project_description}

## Core Features
{chr(10).join(features_list)}

## Technical Architecture
- Frontend: Modern web/mobile framework
- Backend: Scalable API architecture
- Database: Relational or NoSQL database as needed
- Deployment: Cloud-based infrastructure
- Security: Authentication and authorization mechanisms

## Success Metrics
- User engagement and satisfaction
- Performance benchmarks
- Scalability requirements
"""
    st.session_state["prd"] = full_prd

def generate_tasks_from_prd(prd_input):
    st.info("Generating tasks...")
    
    # Extract key information from PRD
    prd_lower = prd_input.lower()
    
    # Generate tasks based on PRD content
    # Create a smart task list based on common project phases and PRD content
    tasks = []
    
    # Phase 1: Planning and Design
    tasks.append("Review and analyze PRD requirements")
    tasks.append("Create technical design document")
    tasks.append("Set up development environment")
    
    # Phase 2: Core Development (based on PRD content)
    if 'frontend' in prd_lower or 'ui' in prd_lower or 'interface' in prd_lower:
        tasks.append("Design and implement user interface")
    if 'backend' in prd_lower or 'api' in prd_lower:
        tasks.append("Develop backend API endpoints")
    if 'database' in prd_lower or 'data' in prd_lower:
        tasks.append("Design and implement database schema")
    if 'authentication' in prd_lower or 'security' in prd_lower:
        tasks.append("Implement authentication and security features")
    
    # Phase 3: Features (extract from PRD)
    if 'feature' in prd_input:
        # Try to extract feature names
        lines = prd_input.split('\n')
        for line in lines:
            if '- ' in line and 'feature' in line.lower():
                feature = line.split('- ', 1)[1].strip()
                if len(feature) < 100:  # Reasonable length
                    tasks.append(f"Implement {feature}")
    
    # Phase 4: Testing and Deployment
    tasks.append("Write unit tests and integration tests")
    tasks.append("Perform code review and quality assurance")
    tasks.append("Deploy application to staging environment")
    tasks.append("Perform user acceptance testing")
    tasks.append("Deploy to production and monitor")
    
    # Limit to reasonable number of tasks
    tasks = tasks[:15]
    
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
