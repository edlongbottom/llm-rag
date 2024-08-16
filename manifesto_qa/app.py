import os
import warnings
import streamlit as st

from dotenv import load_dotenv, find_dotenv

from manifesto_qa.vectordb import VectorDB
from manifesto_qa.document_loader import load_all_pdf_docs
from manifesto_qa.models import get_llm
from manifesto_qa.retrievers import get_retriever, get_self_query_retriever
from manifesto_qa.rag_memory_chain import (
    get_rag_chain_with_memory,
    ask_question_with_history,
)
from manifesto_qa.rag_qa_chain import (
    get_rag_chain,
    get_rag_chain_with_sources,
    ask_question,
    ask_question_with_sources,
)

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

warnings.filterwarnings("ignore")

TEXT_EMBEDDINGS_MODEL = "text2vec-openai"  # "text-embedding-ada-002"
SELF_QUERY_MODEL = "gpt-3.5-turbo-instruct"
GENERATIVE_MODEL = "gpt-3.5-turbo"

WEAVIATE_INDEX_NAME = "ManifestoQa"
WEAVIATE_TEXT_KEY = "text"

DATA_DIR = "/Users/longbe01/Documents/projects/llm-rag/data"
RESET_DB_ON_START = False

_ = load_dotenv(find_dotenv())

vector_db = VectorDB(WEAVIATE_INDEX_NAME, TEXT_EMBEDDINGS_MODEL, GENERATIVE_MODEL)
if RESET_DB_ON_START:
    vector_db.reset_manifesto_schema()
    load_all_pdf_docs(vector_db.instance, DATA_DIR)

llm = get_llm(GENERATIVE_MODEL, os.getenv("OPENAI_API_KEY"))

basic_retriever = get_retriever(vector_db.instance)
retriever = get_self_query_retriever(
    vector_db.instance, llm=SELF_QUERY_MODEL, search_type="similarity", k=5
)

basic_rag_chain = get_rag_chain(llm, retriever)
basic_rag_with_sources = get_rag_chain_with_sources(llm, retriever)
rag_chain = get_rag_chain_with_memory(
    llm, retriever, StreamlitChatMessageHistory(key="chat_history")
)

st.title("General Election 2024 Party Manifesto Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # completion = ask_question(basic_rag_chain, prompt)
        # completion = ask_question_with_sources(basic_rag_with_sources, prompt)
        completion = ask_question_with_history(rag_chain, prompt)
        response = st.write(completion)

    st.session_state.messages.append({"role": "assistant", "content": completion})
