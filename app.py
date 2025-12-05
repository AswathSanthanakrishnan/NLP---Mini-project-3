
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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Project & PRD", "Task Generation", "Employees & Assignments", "Email Reports"]
    )

    # --- Project & PRD Tab ---
    with tab1:
        st.header("📋 Project & PRD")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            default_projects = {
                "Select a project": "",
                "AI-Powered Customer Support Chatbot": "Develop a chatbot that can answer customer questions and resolve common issues. The chatbot should be able to understand natural language and provide personalized responses.",
                "E-commerce Website Redesign": "Redesign an e-commerce website to improve the user experience and increase sales. The project will involve updating the UI, improving navigation, and adding new features.",
                "Mobile App for Task Management": "Create a mobile app that helps users organize their tasks and stay productive. The app should have features like task creation, deadlines, and reminders.",
                "Create New Project": "",
            }
            project_choice = st.selectbox(
                "Choose a default project or create new", 
                list(default_projects.keys()),
                key="project_select"
            )
        
        with col2:
            if project_choice == "Create New Project":
                st.info("💡 Fill in the fields below to create your custom project")
        
        project_name = st.text_input(
            "📝 Project Name",
            value=project_choice if project_choice != "Select a project" and project_choice != "Create New Project" else "",
            placeholder="Enter your project name here...",
        )
        project_description = st.text_area(
            "📄 Project Description",
            value=default_projects[project_choice]
            if project_choice != "Select a project" and project_choice != "Create New Project"
            else "",
            height=200,
            placeholder="Describe your project in detail. Include features, goals, and requirements...",
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🚀 Generate PRD", type="primary", use_container_width=True):
                if not project_name or not project_description:
                    st.error("⚠️ Please enter a project name and description.")
                else:
                    generate_prd(project_name, project_description)

        if "prd" in st.session_state:
            st.markdown("---")
            st.subheader("✨ Generated PRD")
            with st.container():
                st.markdown(st.session_state["prd"])

    # --- Task Generation Tab ---
    with tab2:
        st.header("✅ Task Generation")
        st.markdown("---")
        
        prd_input = st.text_area(
            "📋 PRD Document", 
            value=st.session_state.get("prd", ""), 
            height=400,
            placeholder="Your PRD will appear here automatically after generation, or paste a PRD manually...",
            help="The PRD from the previous tab is automatically loaded here. You can also paste a custom PRD."
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🎯 Generate Tasks", type="primary", use_container_width=True):
                if not prd_input:
                    st.error("⚠️ Please paste the PRD in the text area or generate one in the Project & PRD tab.")
                else:
                    generate_tasks_from_prd(prd_input)

        if "tasks" in st.session_state:
            st.markdown("---")
            st.subheader("📝 Generated Tasks")
            tasks = st.session_state["tasks"]
            if isinstance(tasks, list):
                for i, task in enumerate(tasks, 1):
                    st.markdown(f"**{i}.** {task}")
            else:
                st.write(tasks)

    # --- Employees & Assignments Tab ---
    with tab3:
        st.header("👥 Employees & Assignments")
        st.markdown("---")
        
        if "employees_df" in st.session_state:
            st.subheader("📊 Available Employees")
            st.dataframe(st.session_state["employees_df"], use_container_width=True, hide_index=True)

            if "tasks" in st.session_state:
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("🎯 Assign Tasks", type="primary", use_container_width=True):
                        assign_tasks(
                            st.session_state["tasks"], st.session_state["employees_df"]
                        )
            else:
                st.info("ℹ️ Generate tasks first in the 'Task Generation' tab to assign them.")

            if "assignments_df" in st.session_state:
                st.markdown("---")
                st.subheader("✅ Task Assignments")
                st.dataframe(
                    st.session_state["assignments_df"], use_container_width=True, hide_index=True
                )
                csv = st.session_state["assignments_df"].to_csv(index=False)
                st.download_button(
                    "📥 Download Assignments as CSV",
                    csv,
                    "task_assignments.csv",
                    "text/csv",
                    type="primary",
                )
        else:
            st.warning("⚠️ Please upload an employee CSV file in the sidebar.")
            st.info("💡 The CSV should have columns: `name` and `skills`")

    # --- Email Reports Tab (draft only, no sending) ---
    with tab4:
        st.header("📧 Email Reports")
        st.markdown("---")

        if "assignments_df" not in st.session_state:
            st.info("Generate and assign tasks first to draft an email report.")
        else:
            assignments_df = st.session_state["assignments_df"]

            st.subheader("Draft Email")
            st.caption(f"Project: {st.session_state.get('project_name', 'Not set')}")

            col_from, col_to = st.columns(2)
            with col_from:
                from_email = st.text_input(
                    "From",
                    value=st.session_state.get("email_from", "santhanakrishnan@cua.edu"),
                    help="Sender email address",
                )
            with col_to:
                to_email = st.text_input(
                    "To",
                    value=st.session_state.get("email_to", "aswath.mcs@gmail.com"),
                    help="Recipient email address",
                )

            subject = st.text_input(
                "Subject",
                value=st.session_state.get("email_subject", "Task Assignment Report"),
            )
            signature = st.text_area(
                "Signature / Company footer",
                value=st.session_state.get(
                    "email_signature", "Best regards,\nTasker.ai Team"
                ),
                height=80,
                help="Include your company name or personal sign-off here.",
            )

            col_gen, col_preview = st.columns([1, 2])
            with col_gen:
                if st.button("🪄 Generate Email Draft", type="primary", use_container_width=True):
                    email_body = generate_email_report(
                        assignments_df,
                        from_email,
                        to_email,
                        signature,
                        st.session_state.get("project_name", "Project"),
                        st.session_state.get("prd", ""),
                        st.session_state.get("tasks", []),
                    )
                    if email_body:
                        st.session_state["email_body"] = email_body
                        st.session_state["email_from"] = from_email
                        st.session_state["email_to"] = to_email
                        st.session_state["email_subject"] = subject
                        st.session_state["email_signature"] = signature
                    else:
                        st.error("Could not generate email. Please try again.")

            email_body = st.text_area(
                "Email Body",
                value=st.session_state.get("email_body", ""),
                height=260,
                help="Edit the draft before sending from your email client.",
            )
            st.session_state["email_body"] = email_body

            with st.expander("📄 Assignment summary used for the email"):
                st.dataframe(assignments_df, use_container_width=True, hide_index=True)

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
        formatted_prompt = (prompt or "").strip() + "\n\n"
        # Keep prompt within a reasonable window to avoid model length errors
        formatted_prompt = formatted_prompt[-3500:]
        
        # Generate text
        results = generator(
            formatted_prompt,
            max_new_tokens=220,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
            return_full_text=True,
        )
        
        generated_text = results[0].get("generated_text", "")
        
        # Remove the original prompt from the generated text
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        return generated_text
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return None


def generate_prd(project_name, project_description):
    with st.spinner("🤖 Generating comprehensive PRD document..."):
        # Generate multiple sections using the model
        # Overview section
        overview_prompt = f"Product Requirements Document for {project_name}. Overview: {project_description}"
        overview_text = generate_from_model(overview_prompt)
        st.session_state["project_name"] = project_name
        st.session_state["project_description"] = project_description
        
        # Features section - extract from description
        features_prompt = f"Key features and functionalities for {project_name}: {project_description}"
        features_text = generate_from_model(features_prompt)
        
        # Tools and technologies section
        tools_prompt = f"Technologies, tools, and frameworks needed for {project_name}: {project_description}"
        tools_text = generate_from_model(tools_prompt)
        
        # Parse generated content
        def extract_bullet_points(text, max_items=8):
            """Extract meaningful bullet points from generated text"""
            if not text:
                return []
            items = []
            # Split by sentences and common separators
            sentences = text.replace('\n', ' ').split('.')
            for sent in sentences:
                sent = sent.strip()
                # Filter meaningful sentences
                if len(sent) > 15 and len(sent) < 150:
                    # Remove common prefixes
                    for prefix in ['The', 'This', 'It', 'A', 'An']:
                        if sent.startswith(prefix + ' '):
                            sent = sent[len(prefix) + 1:].strip()
                            break
                    if sent and sent[0].isupper():
                        items.append(sent)
                if len(items) >= max_items:
                    break
            return items
        
        # Extract features from description and generated text
        features_list = []
        desc_lower = project_description.lower()
        
        # Extract explicit features mentioned in description
        desc_sentences = project_description.split('.')
        for sent in desc_sentences:
            sent = sent.strip()
            if any(keyword in sent.lower() for keyword in ['feature', 'should', 'must', 'need', 'include', 'have']):
                if len(sent) > 10 and len(sent) < 200:
                    features_list.append(sent)
        
        # Add generated features
        if features_text:
            gen_features = extract_bullet_points(features_text, max_items=6)
            features_list.extend(gen_features)
        
        # Remove duplicates and limit
        seen = set()
        unique_features = []
        for feat in features_list:
            feat_lower = feat.lower()[:50]  # Use first 50 chars for comparison
            if feat_lower not in seen:
                seen.add(feat_lower)
                unique_features.append(feat)
        features_list = unique_features[:10]
        
        # If still no features, create intelligent defaults based on keywords
        if not features_list:
            if 'chatbot' in desc_lower or 'ai' in desc_lower:
                features_list = [
                    "Natural language processing and understanding capabilities",
                    "Conversational user interface with context awareness",
                    "Integration with knowledge base and FAQ system",
                    "Multi-channel support (web, mobile, API)"
                ]
            elif 'e-commerce' in desc_lower or 'shopping' in desc_lower:
                features_list = [
                    "Product catalog with search and filtering",
                    "Shopping cart and checkout system",
                    "Secure payment gateway integration",
                    "Order management and tracking system"
                ]
            elif 'ios' in desc_lower or 'iphone' in desc_lower or 'ipad' in desc_lower:
                features_list = [
                    "Native iOS application with Swift/SwiftUI",
                    "iOS Human Interface Guidelines compliance",
                    "App Store integration and submission",
                    "Core Data or CloudKit for data persistence",
                    "Push notifications via APNs"
                ]
            elif 'android' in desc_lower and ('ios' not in desc_lower and 'iphone' not in desc_lower):
                features_list = [
                    "Native Android application with Kotlin/Java",
                    "Material Design guidelines compliance",
                    "Google Play Store integration",
                    "Room or SQLite for local database",
                    "Firebase Cloud Messaging for push notifications"
                ]
            elif 'mobile' in desc_lower or 'app' in desc_lower:
                # Check if it's cross-platform or native
                if 'native' in desc_lower or 'swift' in desc_lower or 'kotlin' in desc_lower:
                    # Native development
                    if 'ios' in desc_lower or 'iphone' in desc_lower:
                        features_list = [
                            "Native iOS application with Swift/SwiftUI",
                            "iOS Human Interface Guidelines compliance",
                            "App Store integration",
                            "Core Data for local storage"
                        ]
                    else:
                        features_list = [
                            "Native mobile application",
                            "Platform-specific UI/UX",
                            "App store integration",
                            "Local data persistence"
                        ]
                else:
                    # Cross-platform
                    features_list = [
                        "Cross-platform mobile application (iOS/Android)",
                        "Offline functionality and data synchronization",
                        "Push notifications for updates and reminders",
                        "User authentication and profile management"
                    ]
            else:
                features_list = [
                    "User-friendly and intuitive interface",
                    "Core functionality as per requirements",
                    "Data management and storage system",
                    "Security and authentication mechanisms"
                ]
        
        # Extract tools and technologies
        tools_list = []
        if tools_text:
            tools_list = extract_bullet_points(tools_text, max_items=8)
        
        # Intelligent tool detection based on project type - CHECK iOS/Android FIRST
        if not tools_list:
            # iOS Native Development
            if 'ios' in desc_lower or 'iphone' in desc_lower or 'ipad' in desc_lower or ('native' in desc_lower and 'ios' in project_name.lower()):
                tools_list = [
                    "Xcode - Apple's integrated development environment (IDE)",
                    "Swift programming language for iOS development",
                    "SwiftUI or UIKit for user interface development",
                    "Core Data or CloudKit for data persistence",
                    "CocoaPods or Swift Package Manager for dependency management",
                    "TestFlight for beta testing",
                    "App Store Connect for app distribution"
                ]
            # Android Native Development
            elif 'android' in desc_lower and ('ios' not in desc_lower and 'iphone' not in desc_lower) or ('native' in desc_lower and 'android' in project_name.lower()):
                tools_list = [
                    "Android Studio - Official Android IDE",
                    "Kotlin or Java programming language",
                    "Jetpack Compose or XML layouts for UI",
                    "Room or SQLite for local database",
                    "Gradle for build automation and dependency management",
                    "Google Play Console for app distribution",
                    "Firebase for backend services (optional)"
                ]
            # Cross-platform Mobile (React Native/Flutter)
            elif ('react native' in desc_lower or 'flutter' in desc_lower or 'cross-platform' in desc_lower) and ('native' not in desc_lower):
                tools_list = [
                    "React Native or Flutter for cross-platform development",
                    "Firebase or AWS for backend services",
                    "SQLite or Realm for local database",
                    "RESTful API for server communication"
                ]
            # Generic Mobile App (assume cross-platform if not specified)
            elif 'mobile' in desc_lower or 'app' in desc_lower:
                # Check for native keywords
                if 'native' in desc_lower or 'swift' in desc_lower or 'xcode' in desc_lower:
                    tools_list = [
                        "Xcode and Swift for iOS development",
                        "SwiftUI or UIKit framework",
                        "Core Data for local storage",
                        "App Store Connect for distribution"
                    ]
                elif 'kotlin' in desc_lower or 'android studio' in desc_lower:
                    tools_list = [
                        "Android Studio and Kotlin for Android development",
                        "Jetpack Compose or XML layouts",
                        "Room database for local storage",
                        "Google Play Console for distribution"
                    ]
                else:
                    tools_list = [
                        "React Native or Flutter for cross-platform development",
                        "Firebase or AWS for backend services",
                        "SQLite or Realm for local database",
                        "RESTful API for server communication"
                    ]
            elif 'web' in desc_lower or 'website' in desc_lower:
                tools_list = [
                    "React.js or Vue.js for frontend framework",
                    "Node.js or Python Django/Flask for backend",
                    "PostgreSQL or MongoDB for database",
                    "Docker for containerization and deployment"
                ]
            elif 'ai' in desc_lower or 'chatbot' in desc_lower:
                tools_list = [
                    "Python with TensorFlow or PyTorch for ML models",
                    "NLTK or spaCy for NLP processing",
                    "FastAPI or Flask for API development",
                    "Vector database (Pinecone/Weaviate) for embeddings"
                ]
            else:
                tools_list = [
                    "Modern web framework (React/Vue/Angular)",
                    "Backend API framework (Node.js/Python/Java)",
                    "Database system (PostgreSQL/MySQL/MongoDB)",
                    "Cloud hosting platform (AWS/Azure/GCP)"
                ]
        
        # Build comprehensive PRD
        full_prd = f"""# Product Requirements Document: {project_name}

## 1. Overview
{project_description}

{overview_text[:300] if overview_text else ''}

## 2. Core Features
{chr(10).join(f'- {feat}' for feat in features_list)}

## 3. Technical Requirements

### Tools & Technologies
{chr(10).join(f'- {tool}' for tool in tools_list)}

### Architecture
- **Frontend**: Modern framework based on project requirements
- **Backend**: Scalable API architecture
- **Database**: Appropriate database solution (relational or NoSQL)
- **Deployment**: Cloud-based infrastructure with CI/CD pipeline
- **Security**: Authentication, authorization, and data encryption

## 4. Success Metrics
- User engagement and satisfaction rates
- Performance benchmarks (response time, uptime)
- Scalability and load handling capabilities
- Feature adoption and usage analytics

## 5. Timeline & Milestones
- Phase 1: Planning and Design
- Phase 2: Core Development
- Phase 3: Testing and Quality Assurance
- Phase 4: Deployment and Monitoring
"""
        st.session_state["prd"] = full_prd
        st.success("✅ PRD generated successfully!")

def generate_tasks_from_prd(prd_input):
    with st.spinner("🤖 Generating comprehensive task list from PRD..."):
        # Generate tasks using model based on PRD content
        task_prompt = f"Generate a detailed task list for this project PRD:\n\n{prd_input[:1000]}\n\nTasks:"
        tasks_text = generate_from_model(task_prompt)
        
        tasks = []
        prd_lower = prd_input.lower()
        project_name_lower = prd_lower.split('product requirements document:')[1].split('\n')[0].strip().lower() if 'product requirements document:' in prd_lower else ""
        
        # Detect platform type
        is_ios = 'ios' in prd_lower or 'iphone' in prd_lower or 'ipad' in prd_lower or 'xcode' in prd_lower or 'swift' in prd_lower or 'ios' in project_name_lower
        is_android = ('android' in prd_lower or 'kotlin' in prd_lower or 'android studio' in prd_lower) and not is_ios
        is_native_mobile = is_ios or is_android
        is_cross_platform = ('react native' in prd_lower or 'flutter' in prd_lower or 'cross-platform' in prd_lower) and not is_native_mobile
        
        # Phase 1: Planning and Setup
        tasks.append("Review and analyze PRD requirements thoroughly")
        tasks.append("Create detailed technical design document")
        
        # Platform-specific setup tasks
        if is_ios:
            tasks.append("Install and configure Xcode development environment")
            tasks.append("Set up Apple Developer account and certificates")
            tasks.append("Create new Xcode project with Swift/SwiftUI")
            tasks.append("Configure project settings (bundle ID, version, etc.)")
        elif is_android:
            tasks.append("Install and configure Android Studio")
            tasks.append("Set up Android SDK and required tools")
            tasks.append("Create new Android project with Kotlin/Java")
            tasks.append("Configure app manifest and build.gradle")
        elif is_cross_platform:
            tasks.append("Set up React Native or Flutter development environment")
            tasks.append("Initialize cross-platform project structure")
            tasks.append("Configure platform-specific settings")
        else:
            tasks.append("Set up development environment and tools")
        
        tasks.append("Initialize project repository and version control")
        
        # Extract features from PRD and create tasks
        lines = prd_input.split('\n')
        feature_section = False
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'feature' in line_lower and ('##' in line or '###' in line):
                feature_section = True
                continue
            if feature_section and ('##' in line or '###' in line) and 'feature' not in line_lower:
                feature_section = False
            if feature_section and '- ' in line:
                feature = line.split('- ', 1)[1].strip()
                if len(feature) > 5 and len(feature) < 150:
                    tasks.append(f"Implement feature: {feature}")
        
        # Extract tools and create setup tasks
        tools_section = False
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'tool' in line_lower or 'technolog' in line_lower:
                tools_section = True
                continue
            if tools_section and ('##' in line or '###' in line):
                tools_section = False
            if tools_section and '- ' in line:
                tool = line.split('- ', 1)[1].strip()
                if len(tool) > 5:
                    # Skip generic setup if it's iOS/Android specific
                    if is_ios and 'xcode' in tool.lower():
                        continue  # Already added above
                    if is_android and 'android studio' in tool.lower():
                        continue  # Already added above
                    tasks.append(f"Set up and configure {tool}")
        
        # Platform-specific development tasks
        if is_ios:
            tasks.append("Design iOS UI/UX following Human Interface Guidelines")
            tasks.append("Implement SwiftUI views or UIKit components")
            tasks.append("Set up Core Data or CloudKit for data persistence")
            tasks.append("Configure App Store Connect and app metadata")
            tasks.append("Implement push notifications using APNs")
            tasks.append("Add app icons and launch screens for all device sizes")
        elif is_android:
            tasks.append("Design Android UI/UX following Material Design guidelines")
            tasks.append("Implement Jetpack Compose or XML layouts")
            tasks.append("Set up Room database or SQLite for local storage")
            tasks.append("Configure Google Play Console and app listing")
            tasks.append("Implement Firebase Cloud Messaging for push notifications")
            tasks.append("Add app icons and adaptive icons for different densities")
        else:
            # Core development tasks based on PRD content (web/cross-platform)
            if 'frontend' in prd_lower or 'ui' in prd_lower or 'interface' in prd_lower:
                tasks.append("Design user interface mockups and wireframes")
                tasks.append("Implement responsive frontend components")
                tasks.append("Integrate frontend with backend APIs")
        
        if 'backend' in prd_lower or 'api' in prd_lower:
            tasks.append("Design and develop RESTful API endpoints")
            tasks.append("Implement API authentication and authorization")
            tasks.append("Create API documentation")
        
        if 'database' in prd_lower or 'data' in prd_lower:
            tasks.append("Design database schema and relationships")
            tasks.append("Implement database migrations")
            tasks.append("Set up database indexing and optimization")
        
        if 'authentication' in prd_lower or 'security' in prd_lower:
            tasks.append("Implement user authentication system")
            tasks.append("Add security measures and data encryption")
            tasks.append("Set up role-based access control")
        
        # Parse generated tasks from model
        if tasks_text:
            gen_lines = tasks_text.split('\n')
            for line in gen_lines:
                line = line.strip()
                if line:
                    # Remove numbering
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    if len(line) > 10 and len(line) < 200:
                        # Avoid duplicates
                        if not any(line.lower() in existing.lower() or existing.lower() in line.lower() for existing in tasks):
                            tasks.append(line)
        
        # Testing and deployment phase - platform specific
        tasks.append("Write comprehensive unit tests")
        tasks.append("Implement integration tests")
        
        if is_ios:
            tasks.append("Test on iOS Simulator and physical devices")
            tasks.append("Configure TestFlight for beta testing")
            tasks.append("Submit app for App Store review")
            tasks.append("Set up App Store analytics and crash reporting")
        elif is_android:
            tasks.append("Test on Android emulator and physical devices")
            tasks.append("Set up internal testing track in Google Play Console")
            tasks.append("Submit app for Google Play Store review")
            tasks.append("Configure Google Play Console analytics")
        else:
            tasks.append("Perform code review and refactoring")
            tasks.append("Set up CI/CD pipeline")
            tasks.append("Deploy to staging environment")
            tasks.append("Perform user acceptance testing (UAT)")
            tasks.append("Deploy to production environment")
            tasks.append("Set up monitoring and logging")
        
        tasks.append("Create user documentation and guides")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tasks = []
        for task in tasks:
            task_lower = task.lower()[:60]  # Use first 60 chars for comparison
            if task_lower not in seen:
                seen.add(task_lower)
                unique_tasks.append(task)
        
        # Limit to reasonable number but keep important ones
        if len(unique_tasks) > 25:
            # Keep first 5 (planning), middle tasks (development), and last 5 (deployment)
            unique_tasks = unique_tasks[:5] + unique_tasks[5:-5][:15] + unique_tasks[-5:]
        
        st.session_state["tasks"] = unique_tasks
        st.success(f"✅ Generated {len(unique_tasks)} tasks successfully!")

def format_assignments_summary(assignments_df):
    """Create a concise summary of task->assignee mapping."""
    lines = []
    for _, row in assignments_df.iterrows():
        task = row.get("Task", "")
        assignee = row.get("Assigned To", "")
        skills = row.get("Skills", "")
        lines.append(f"- {task} -> {assignee} (Skills: {skills})")
    return "\n".join(lines)

def generate_email_report(assignments_df, from_email, to_email, signature, project_name, prd_text, tasks_list):
    """Use the local text generator to craft a concise status email, anchored on real assignment data."""
    summary = format_assignments_summary(assignments_df)
    task_lines = "\n".join(f"- {t}" for t in tasks_list[:12]) if tasks_list else ""
    prd_excerpt = (prd_text or "")[:400]

    prompt = (
        "Write a short, professional status email. Keep it concise and client-ready.\n"
        f"Project name: {project_name}\n"
        f"From: {from_email}\n"
        f"To: {to_email}\n"
        "Structure:\n"
        "1) Greeting and one-sentence status summary about assignments being created.\n"
        "2) Bullet list EXACTLY using the provided assignments list (do not invent or repeat). Keep them brief.\n"
        "3) One sentence tying back to scope/context from the PRD excerpt.\n"
        "4) Closing line that invites follow-up.\n"
        f"Signature block (use as-is):\n{signature}\n"
        "Assignments list (use these bullets verbatim, do not add new items):\n"
        f"{summary}\n"
        f"Tasks (raw list, optional to mention count):\n{task_lines}\n"
        f"PRD excerpt:\n{prd_excerpt}\n"
        "Return only the email body, no subject line."
    )

    generated = generate_from_model(prompt) or ""

    # If the model output lacks our required bullets, fall back to a deterministic version
    if summary not in generated:
        deterministic = (
            f"Hello,\n\nHere is the latest assignment update for {project_name}:\n"
            f"{summary}\n\n"
            f"Scope reference: {prd_excerpt[:200]}...\n\n"
            f"Please let me know if you need any changes or additional details.\n\n{signature}"
        )
        return deterministic

    return generated.strip()

def assign_tasks(tasks, employees_df):
    with st.spinner("🤖 Matching tasks to employees based on skills..."):
        assignments = []
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            employee_skills = employees_df["skills"].tolist()
            skill_embeddings = model.encode(employee_skills, convert_to_tensor=True)

            for task in tasks:
                task_embedding = model.encode(task, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(task_embedding, skill_embeddings)
                
                # Convert tensor to CPU and then to numpy to avoid device error
                cosine_scores_cpu = cosine_scores.cpu().detach().numpy()
                best_employee_idx = np.argmax(cosine_scores_cpu)
                confidence_score = float(cosine_scores_cpu[0][best_employee_idx])
                
                assignments.append(
                    {
                        "Task": task,
                        "Assigned To": employees_df.iloc[best_employee_idx]["name"],
                        "Skills": employees_df.iloc[best_employee_idx]["skills"],
                        "Confidence": f"{confidence_score:.2%}",
                    }
                )
            assignments_df = pd.DataFrame(assignments)
            st.session_state["assignments_df"] = assignments_df
            st.success(f"✅ Successfully assigned {len(assignments)} tasks!")
        except Exception as e:
            st.error(f"❌ Error assigning tasks: {e}")
            import traceback
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
