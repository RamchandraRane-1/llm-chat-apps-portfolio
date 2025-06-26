import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

# Session state setup
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''

# Page config
st.set_page_config(page_title="Chat GPT Clone")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h1>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.title("Settings")
st.session_state['API_Key'] = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

role = st.sidebar.selectbox("Select Assistant Role", [
    "Medical Doctor",
    "PhD in Computer Science",
    "PhD in Biology",
    "PhD in Physics",
    "AI Researcher",
    "Legal Expert"
])

summarise_button = st.sidebar.button("Summarise the conversation", key="summarise")
if summarise_button and st.session_state['conversation']:
    st.sidebar.write("Nice chatting with you my friend:\n\n" + st.session_state['conversation'].memory.buffer)


# Response function
def getresponse(userInput, api_key):
    if st.session_state['conversation'] is None:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo'
        )

        system_prompt = f"You are a highly knowledgeable assistant with expertise as a {role}. Answer user questions thoroughly and clearly, just as a domain expert would."

        memory = ConversationBufferMemory(return_messages=True)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        st.session_state['conversation'] = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )

    response = st.session_state['conversation'].predict(input=userInput)
    return response



# Chat UI
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input.strip() != "":
            st.session_state['messages'].append(user_input)
            model_response = getresponse(user_input, st.session_state['API_Key'])
            st.session_state['messages'].append(model_response)

            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if i % 2 == 0:
                        message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')
