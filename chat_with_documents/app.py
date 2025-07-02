import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import altair as alt
import os
import tempfile
import logging
import time
import openai

# Logging setup
logging.basicConfig(filename='chatbot_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None

# Page config
st.set_page_config(page_title="Chatbot with Document Retrieval")
st.markdown("<h1 style='text-align: center;'> Chatbot </h1>", unsafe_allow_html=True)

# Sidebar: API Key and file upload
st.sidebar.title("Configuration")
st.session_state['API_Key'] = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt", "docx", "csv", "xlsx"])

# Initialize document retriever or load CSV/Excel
retriever = None
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if file_extension in ["csv", "xlsx"]:
        if file_extension == "csv":
            df = pd.read_csv(tmp_file_path)
        else:
            df = pd.read_excel(tmp_file_path)
        st.session_state['raw_df'] = df
        st.write("### Uploaded Data Preview:")
        st.dataframe(df)
    else:
        if file_extension == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = UnstructuredFileLoader(tmp_file_path)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['API_Key'])
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=st.session_state['API_Key'],
            model_name="gpt-3.5-turbo"
        )

        st.session_state['conversation'] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

# Retry mechanism
def robust_invoke(llm, prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            logging.error(f"OpenAI API Error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"OpenAI API error: {e}"

# Get chatbot response
@st.cache_data(show_spinner=False)
def get_response(user_input):
    conversation = st.session_state['conversation']
    chat_history = st.session_state['chat_history']
    try:
        result = conversation({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result["answer"]))
        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {result['answer']}")
        return result["answer"]
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return f"Error during response generation: {e}"

# UI Input + Response
response_container = st.container()
container = st.container()

with container:
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area("Ask a question:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            st.session_state['messages'].append(user_input)
            logging.info(f"User input: {user_input}")
            if st.session_state['conversation']:
                model_response = get_response(user_input)
            elif st.session_state['raw_df'] is not None:
                df = st.session_state['raw_df']
                llm = ChatOpenAI(
                    temperature=0,
                    openai_api_key=st.session_state['API_Key'],
                    model_name="gpt-3.5-turbo"
                )
                context = f"This is a dataframe with {len(df)} rows and these columns: {', '.join(df.columns)}."
                sample_data = df.head(5).to_dict(orient="records")
                context += f" Here are the first 5 rows: {sample_data}."
                prompt = (
                    f"Given this dataset context:\n{context}\n\n"
                    f"Now answer the user's question in a helpful and well-explained manner:\n{user_input}"
                )
                model_response = robust_invoke(llm, prompt)
            else:
                model_response = "Please upload a valid file and provide an API key."

            st.session_state['messages'].append(model_response)
            logging.info(f"Model response: {model_response}")

with response_container:
    for i, msg in enumerate(st.session_state['messages']):
        is_user = (i % 2 == 0)
        message(msg, is_user=is_user, key=f"msg_{i}")

# Sidebar summary
if st.sidebar.button("Summarize Conversation"):
    full_chat = "\n".join([f"User: {q}\nAI: {a}" for q, a in st.session_state['chat_history']])
    st.sidebar.markdown("### Chat Summary")
    st.sidebar.markdown(full_chat.replace("\n", "  \n"))
