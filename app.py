

import httpx
import os

def test_openai_connectivity():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY is missing")
        return

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = httpx.get("https://api.openai.com/v1/models", headers=headers)
        print("✅ OpenAI API reachable, status code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("❌ Could not reach OpenAI API:", e)

# Call this once during startup
test_openai_connectivity()

import openai
import langchain
import langchain_openai
import sys

print("OPENAI VERSION:", openai.__version__)
print("LANGCHAIN VERSION:", langchain.__version__)
print("PYTHON VERSION:", sys.version)

import logging
logging.basicConfig(level=logging.DEBUG)
import streamlit as st
from langchain_openai import ChatOpenAI


import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Function to return the response
def load_answer(question):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    return llm.invoke(question)


#App UI starts here
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("Chatbot")

#Gets the user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input=get_text()
response = load_answer(user_input)

submit = st.button('Generate')  

#If generate button is clicked
if submit:

    st.subheader("Answer:")

    st.write(response.content)

