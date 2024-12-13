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
    pm = autogen.AssistantAgent(name="Product_manager", system_message="Creative in software product ideas.", llm_config=llm_config)
    pjm = autogen.AssistantAgent(name="Project_manager", system_message="I need everyone in my team to work faster", llm_config=llm_config)

    # Initialize GroupChat and Manager
    groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm, pjm], messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Main container for chat
    with st.container():
        user_input = st.text_input("Send a Message to the Group Chat Agent")
        
        if user_input:
            async def process_group_chat():
                response = await manager.a_step(user_proxy_name="User_proxy", user_message=user_input)
                # Display chat responses
                for message in response:
                    with st.chat_message(message['sender']):
                        st.markdown(message['content'])

            # Run asynchronous group chat processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_group_chat())

else:
    st.warning("Please provide API configuration details in the sidebar.", icon="⚠️")
