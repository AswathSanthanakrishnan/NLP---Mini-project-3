# 📝 tasker.ai Summary

## 🎯 Project Goal and Motivation
The goal of this project was to create an intelligent, AI-powered project management and task delegation platform called **tasker.ai**.

The motivation was to explore the practical application of NLP to a complex, real-world optimization problem, contributing a novel idea and a functional prototype. This project was undertaken as a learning exercise to meet the requirements of the CSC 446 Final Project.

## 🛠️ Methods
The application was built as a single Python script using the Streamlit framework. The core NLP functionalities are:

*   **🤖 Task Generation:** A generative AI model (`gemini-pro`) is used to generate a list of tasks from a project description.
*   **🧠 Task/Skill Matching:** The `sentence-transformers` library is used to compute the semantic similarity between task descriptions and employee skills. A pre-trained model (`all-MiniLM-L6-v2`) is used to create embeddings for tasks and skills. The cosine similarity between these embeddings is used to find the best employee for each task.

## ✨ Results
The resulting application is a functional prototype that allows users to:

*   📤 Upload employee data (CSV).
*   📁 Define projects with a name and description.
*   ģ Generate tasks for a project using AI.
*   🧑‍💻 Intelligently assign tasks to employees based on their skills.
*   📥 Download a report of the task assignments.

The application demonstrates the successful integration of NLP models to automate a key aspect of project management.

## 🤔 Reflections on Foresight and Magnanimity
###  foresight
The project successfully applied the virtue of foresight by focusing on the core goal of intelligent task assignment. Technical decisions were made to support this goal, such as using semantic similarity for matching and scoping the project to a single-file Streamlit application. This ensured the delivery of a functional and demonstrable prototype.

### magnanimity
The project embraced the challenge of modeling a complex human process. By using advanced NLP techniques to reason about the relationships between project needs and individual skills, the project team stretched its abilities to create a high-value system. This was done in the spirit of growth and learning, contributing a novel idea to the field of AI-powered project management.

