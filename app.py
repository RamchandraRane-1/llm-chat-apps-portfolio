
import streamlit as st
from langchain_openai import ChatOpenAI
import os

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY environment variable not found.")
    st.stop()

# From here down is all the StreamLit UI
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("Hey, I'm your Chat GPT")



if "sessionMessages" not in st.session_state:
     st.session_state.sessionMessages = [
        SystemMessage(content="You are a helpful assistant.")
    ]



def load_answer(question):

    st.session_state.sessionMessages.append(HumanMessage(content=question))

    assistant_answer  = chat.invoke(st.session_state.sessionMessages)

    st.session_state.sessionMessages.append(AIMessage(content=assistant_answer.content))

    return assistant_answer.content


def get_text():
    input_text = st.text_input("You: ")
    return input_text

chat = ChatOpenAI(
    temperature=0,
    openai_api_key=api_key  
)

user_input=get_text()
submit = st.button('Generate')  

if submit:
    
    response = load_answer(user_input)
    st.subheader("Answer:")
    st.write(response)

