import streamlit as st
import asyncio
import autogen
from autogen import AssistantAgent, UserProxyAgent
from openai import AzureOpenAI
from autogen.coding import LocalCommandLineCodeExecutor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import os
os.environ["AUTOGEN_USE_DOCKER"] = "False"  # Disable Docker usage

# setup page title and description
st.set_page_config(page_title="AutoGen Chat app", page_icon="🤖", layout="wide")

st.markdown(
    "Welcome to the world of Agents. You can use it to chat with AOAI Models. Your wish is my command."
)
st.markdown("An example question you can ask is: 'How is the stock price of Microsoft today? Summarize the news for me.'")
st.markdown("Start by getting your AOAI Model details")

class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

# add placeholders for selected model, key, and base URL
selected_model = None
selected_key = None
api_base_url = None
api_version = None
api_type= None

# setup sidebar: models to choose from, API key input, and base URL input
with st.sidebar:
    st.header("Azure OpenAI Configuration")
    api_version = "2024-05-01-preview"
    api_type= "azure"
    selected_model = st.selectbox("Model", ["gpt4mini", "gpt4o"], index=1)
    selected_key = st.text_input("API Key", type="password")
    api_base_url = st.text_input("API Base URL", placeholder="https://<your-resource-name>.openai.azure.com/")

# setup main area: user input and chat messages
with st.container():
    user_input = st.text_input("User Input")
    # only run if user input is not empty and model, key, and base URL are provided
    if user_input:
        if not selected_key or not selected_model or not api_base_url:
            st.warning("You must provide a valid OpenAI API key, choose a model, and provide the API base URL", icon="⚠️")
            st.stop()
        
        # setup request timeout and config list
        llm_config = {
            # "request_timeout": 600,
            "config_list": [
                {"model": selected_model, 
                 "api_key": selected_key,
                 "base_url": api_base_url,
                 "api_type": api_type,
                 "api_version": api_version},
            ],
            # "seed": "42",  # seed for reproducibility
            "cache_seed": 42,
            "temperature": 0,  # temperature of 0 means deterministic output
        }
         # create an AssistantAgent instance named "assistant"
        assistant = TrackableAssistantAgent(
            name="assistant", llm_config=llm_config)

        # create a UserProxyAgent instance named "user"
        user_proxy = TrackableUserProxyAgent(
            name="user",
            human_input_mode="NEVER", 
            llm_config=llm_config,
            # code_execution_config=False,
            code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")},
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

         # Function to convert chart to base64
        def get_image_base64(fig):
            img_bytes = io.BytesIO()
            fig.savefig(img_bytes, format='PNG')
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            return img_base64
        
        # Define function to initiate chat
        
        async def initiate_chat():
            # Simulate the chat interaction, assistant generates a chart
            message = await assistant.chat(user_input)

             # Display assistant's response in the chat
            with st.chat_message("assistant"):
                st.markdown(response)
                
             # If the assistant generates a chart, display it
            if 'chart' in response:  # Assuming the assistant returns a key 'chart' if it generates one
                chart_data = response['chart']

                # Convert chart data to base64 (assuming it's a Matplotlib figure)
                if isinstance(chart_data, plt.Figure):
                    img_base64 = get_image_base64(chart_data)
                    st.image(f"data:image/png;base64,{img_base64}", caption="Generated Chart", use_column_width=True)

        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())
      
        
        
