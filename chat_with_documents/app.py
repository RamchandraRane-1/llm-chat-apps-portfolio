import streamlit as st
from langchain.schema import Document
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import tempfile
import logging
import time
import openai
from document_parser import process_images_from_file

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(filename='chatbot_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize session state
for key in ['chat_history', 'conversation', 'messages', 'API_Key', 'raw_df']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'chat_history' else None if key == 'conversation' else "" if key == 'API_Key' else []

st.set_page_config(page_title="Chatbot with Document Retrieval")
st.markdown("<h1 style='text-align: center;'>What can I help with?</h1>", unsafe_allow_html=True)

st.sidebar.title("Configuration")
st.session_state['API_Key'] = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents", type=["pdf", "txt", "docx", "csv", "xlsx", "pptx"], accept_multiple_files=True)

retriever = None
texts = []
dataframes = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        captions = []
        file_extension = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if file_extension in ["csv", "xlsx"]:
            df = pd.read_csv(tmp_file_path) if file_extension == "csv" else pd.read_excel(tmp_file_path)
            dataframes.append(df)

        else:
            images, captions = process_images_from_file(tmp_file_path, file_extension)
            loader = PyPDFLoader(tmp_file_path) if file_extension == "pdf" else UnstructuredFileLoader(tmp_file_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)

            if captions and isinstance(captions, list):
                image_text = "\n\n".join([f"Image {i+1}: {cap}" for i, cap in enumerate(captions)])
                captioned_doc = f"""These are auto-generated image captions from the uploaded file:

{image_text}

Use this information to support relevant queries related to the visual content."""
                split_docs.append(Document(page_content=captioned_doc))

            texts.extend(split_docs)

if texts:
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

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    st.session_state['raw_df'] = combined_df
    st.write("### Uploaded Data Preview:")
    st.dataframe(combined_df.head(10))

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

def get_response(user_input):
    conversation = st.session_state['conversation']
    chat_history = st.session_state['chat_history']
    try:
        result = conversation.invoke({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result["answer"]))

        source_docs = result.get('source_documents', [])
        citations = set()
        for doc in source_docs:
            source = doc.metadata.get('source', 'Unknown').split('/')[-1].split('\\')[-1]
            page = doc.metadata.get('page')
            if page is not None:
                citations.add(f"{source}, p. {page + 1}")
            else:
                citations.add(source)

        final_answer = result["answer"]
        if citations:
            final_answer += "\n\n**Sources:**\n- " + "\n- ".join(sorted(list(citations)))

        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {final_answer}")
        return final_answer

    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return f"Error during response generation: {e}"

def query_dataframe(user_input, df, api_key):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo"
    )
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        **{"agent_type": "openai-tools"},
        allow_dangerous_code=True
    )
    return agent.run(user_input)

if __name__ == "__main__":
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_area("Ask a question:", key='input', height=68)
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                st.session_state['messages'].append(user_input)
                logging.info(f"User input: {user_input}")

                if st.session_state['conversation']:
                    model_response = get_response(user_input)
                elif isinstance(st.session_state['raw_df'], pd.DataFrame):
                    df = st.session_state['raw_df']
                    model_response = query_dataframe(user_input, df, st.session_state['API_Key'])
                elif not uploaded_files:
                    model_response = "⚠️ No file uploaded. Please upload a file to get started."
                else:
                    model_response = (
                        "📄 Your uploaded file type was processed as a document.\n\n"
                        "Please ask a question related to the content, and I’ll help retrieve information!"
                    )

                st.session_state['messages'].append(model_response)
                logging.info(f"Model response: {model_response}")

    with response_container:
        for i, msg in enumerate(st.session_state['messages']):
            is_user = (i % 2 == 0)
            message(msg, is_user=is_user, key=f"msg_{i}")

    if st.sidebar.button("Summarize Conversation"):
        full_chat = "\n".join([f"User: {q}\nAI: {a}" for q, a in st.session_state['chat_history']])
        st.sidebar.markdown("### Chat Summary")
        st.sidebar.markdown(full_chat.replace("\n", "  \n"))
