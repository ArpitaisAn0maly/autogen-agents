# AI Agent Framework Showcase  

This repository demonstrates the use of Microsoft's **Autogen Framework** for creating and managing AI agents. It includes multiple Proofs of Concept (POCs) showcasing how to leverage the Autogen framework for building intelligent, collaborative AI agents.  

## Features  

1. **Agent Creation Using OpenAI API**  
   - `app.py` enables you to create AI agents powered by OpenAI's API.  

2. **Multi-Agent Chat with Azure OpenAI**  
   - `AgentApp.py` demonstrates task-solving with code generation, execution, and debugging capabilities.  
   - `GroupAgentApp.py` showcases conversable agents, allowing tools and humans to collaboratively perform tasks through automated multi-agent chats.  

## Highlights  

- **Task Solving with Code Generation**: Utilize AI agents to generate, execute, and debug code.  
- **Multi-Agent Collaboration**: Enable seamless conversations among agents, tools, and humans to collectively solve problems.  
- **Streamlit Integration**: Build interactive web apps for your AI agent use cases.  


## Technologies Used  

- **Python**: Core programming language for development.  
- **Streamlit**: For building interactive web apps.  
- **Microsoft Autogen Framework**: Framework for creating, managing, and orchestrating AI agents.

## Hosting on Streamlit Cloud

You can easily host your Streamlit app on **Streamlit Cloud** by connecting your GitHub repository. Once deployed, your app will be accessible online without the need to run it locally. Visit [Streamlit Cloud](https://streamlit.io/cloud) for more details on how to deploy your app directly from GitHub.


# Project Setup and Instructions

## Installation and Setup

Follow these steps to set up the project and run the Streamlit app.

---

### Install Dependencies

#### Clone the repository:  
   ```bash  
   git clone [(https://github.com/ArpitaisAn0maly/autogen-agents.git)]
   cd autogen-agents

#### For Azure OpenAI setup:
```bash
pip install -r requirements.txt

#### For OpenAI setup:
```bash
pip install -r openaireq.txt

#### For runing Agents locally with OpenAI API:
```bash
streamlit run app.py

#### For running Multi-Agent Chat locally with Azure OpenAI:
```bash
streamlit run AgentApp.py
streamlit run GroupAgentApp.py



