import os
import warnings
import streamlit as st

from dotenv import load_dotenv, find_dotenv

from manifesto_qa.vectordb import VectorDB
from manifesto_qa.document_loader import load_all_pdf_docs
from manifesto_qa.models import get_llm
from manifesto_qa.retrievers import get_self_query_retriever
from manifesto_qa.rag_memory_chain import (
    get_rag_chain_with_memory,
    ask_question_with_history,
)

from langchain_core.runnables.base import Runnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

warnings.filterwarnings("ignore")

### Question-answering models
# OpenAI: "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"
# Cohere: command, command-light, command-r
# Mistral: mistral-large-latest, mistral-medium-latest, mistral-small-latest, open-mistral-nemo

TEXT_EMBEDDINGS_MODEL = "text2vec-openai"  # "text-embedding-ada-002"
SELF_QUERY_MODEL = "gpt-3.5-turbo"  # "gpt-3.5-turbo-instruct"
GENERATIVE_MODEL = "command-r"

WEAVIATE_INDEX_NAME = "ManifestoQa"
WEAVIATE_TEXT_KEY = "text"


def init_vector_database(load_docs: bool = False) -> VectorDB:

    vector_db = VectorDB(WEAVIATE_INDEX_NAME, TEXT_EMBEDDINGS_MODEL, GENERATIVE_MODEL)
    if load_docs:
        vector_db.reset_manifesto_schema()
        load_all_pdf_docs(vector_db.instance, os.getenv("DATA_DIR"))

    return vector_db


def init_rag_chain(vector_db: VectorDB) -> Runnable:
    generative_llm = get_llm(GENERATIVE_MODEL)
    self_query_llm = get_llm(SELF_QUERY_MODEL, temperature=0.0)
    retriever = get_self_query_retriever(
        vector_db.instance, self_query_llm, search_type="similarity", k=5
    )
    rag_chain = get_rag_chain_with_memory(
        generative_llm, retriever, StreamlitChatMessageHistory(key="chat_history")
    )
    return rag_chain


def run():

    _ = load_dotenv(find_dotenv())

    vector_db = init_vector_database()
    rag_chain = init_rag_chain(vector_db)

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
            completion = ask_question_with_history(rag_chain, prompt)
            response = st.write(completion)

        st.session_state.messages.append({"role": "assistant", "content": completion})


run()
