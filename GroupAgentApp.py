import streamlit as st
import asyncio
import autogen
import os

# Set environment variable to disable Docker usage
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# Page configuration
st.set_page_config(page_title="Multi-Agent GroupChat App", page_icon="🤖", layout="wide")

# App header
st.markdown(
    "Welcome to the Multi-Agent GroupChat App. Interact with multiple agents like a Coder, Product Manager, and Project Manager."
)
st.markdown("Start by configuring your AOAI Model details.")

# Sidebar configuration
with st.sidebar:
    st.header("Azure OpenAI Configuration")
    api_version = "2024-05-01-preview"
    api_type = "azure"
    selected_model = st.selectbox("Model", ["gpt4mini", "gpt4o"], index=1)
    selected_key = st.text_input("API Key", type="password")
    api_base_url = st.text_input("API Base URL", placeholder="https://<your-resource-name>.openai.azure.com/")

# Placeholder for user input and group chat
if selected_key and selected_model and api_base_url:
    # Configure LLM settings
    config_list = [
        {
            "model": selected_model,
            "api_key": selected_key,
            "base_url": api_base_url,
            "api_type": api_type,
            "api_version": api_version,
        }
    ]
    llm_config = {"config_list": config_list, "cache_seed": 42}

    # Define agents
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        code_execution_config={
            "last_n_messages": 2,
            "work_dir": "groupchat",
            "use_docker": False,
        },
        human_input_mode="TERMINATE",
    )
    coder = autogen.AssistantAgent(name="Coder", llm_config=llm_config)
    pdm = autogen.AssistantAgent(name="Product_manager", system_message="As a product manager, I focus on delivering valuable and innovative products. My goal is to align the team on customer needs, ensure effective collaboration across departments, and drive product strategy and roadmap execution.", llm_config=llm_config)
    pjm = autogen.AssistantAgent(name="Project_manager", system_message="As a project manager, I need everyone in my team to work faster and meet deadlines efficiently. Prioritize tasks and keep the workflow smooth.", llm_config=llm_config)

    # Initialize GroupChat and Manager
    groupchat = autogen.GroupChat(agents=[user_proxy,coder,pjm], messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Main container for chat
    with st.container():
        user_input = st.text_input("Send a Message to the Group Chat Agent")          

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        
        # Define function to initiate chat
        
        async def initiate_chat():
            await user_proxy.a_initiate_chat(
                manager,
                message=user_input,
        )

            # Update the messages after the chat response
            chat_messages = manager.groupchat.messages  # Get the updated messages from the group chat

            # # Debugging: Print the messages to check their structure
            # print(f"DEBUG: Total messages: {len(chat_messages)}")  # Print the total number of messages
            for idx, msg in enumerate(chat_messages):
                role = msg.get('role', 'Unknown')  # Use .get() to avoid KeyError if 'role' doesn't exist
                content = msg.get('content', 'No content')  # Use .get() to avoid KeyError
                name = msg.get('name', 'Unknown')  # Use .get() to avoid KeyError

            #     # Debugging: Print each message's role, content, and name
            #     print(f"DEBUG: Message {idx} - Role: {role}, Name: {name}, Content: {content}")

            # Display message with role and content clearly
            st.markdown(f"### **{role} ({name}) says:**")
            st.write(f"    {content}")
           

        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())
      


