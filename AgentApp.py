import streamlit as st
import asyncio
from autogen import AssistantAgent, UserProxyAgent
from openai import AzureOpenAI

# setup page title and description
st.set_page_config(page_title="AutoGen Chat app", page_icon="🤖", layout="wide")

st.markdown(
    "Welcome to the world of Agents. You can use it to chat with AOAI Models. Your wish is my command."
)
st.markdown("An example question you can ask is: 'How is the stock price of Microsoft today? Summarize the news for me.'")
st.markdown("Start by getting your AOAI Model details")

# add placeholders for selected model, key, and base URL
selected_model = None
selected_key = None
api_base_url = None

# setup sidebar: models to choose from, API key input, and base URL input
with st.sidebar:
    st.header("OpenAI Configuration")
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
        
        # setup request timeout
