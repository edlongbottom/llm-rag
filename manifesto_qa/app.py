import os

from dotenv import load_dotenv, find_dotenv

from manifesto_qa.client import get_weaviate_client
from manifesto_qa.vector_db import VectorDB
from manifesto_qa.document_loader import load_all_pdf_docs
from manifesto_qa.rag_chain import (
    get_prompt_template,
    get_prompt_template_with_history,
    run_rag_chain,
    run_rag_chain_with_sources,
    run_rag_chain_with_history,
    get_llm,
    get_retriever,
    get_self_query_retriever,
)

TEXT_EMBEDDINGS_MODEL = "text2vec-openai"  # "text-embedding-ada-002"
SELF_QUERY_MODEL = "gpt-3.5-turbo-instruct"
GENERATIVE_MODEL = "gpt-3.5-turbo"

WEAVIATE_INDEX_NAME = "ManifestoQa"
WEAVIATE_TEXT_KEY = "text"

DATA_DIR = "/Users/longbe01/Documents/projects/llm-rag/data"
RESET_DB_ON_START = False

_ = load_dotenv(find_dotenv())
weaviate_client, _ = get_weaviate_client()

vector_db = VectorDB(
    weaviate_client, WEAVIATE_INDEX_NAME, TEXT_EMBEDDINGS_MODEL, GENERATIVE_MODEL
)
if RESET_DB_ON_START:
    vector_db.reset_manifesto_schema()
    load_all_pdf_docs(vector_db.instance, DATA_DIR)

# retriever = get_retriever(vector_db.instance, search_type="similarity", k=5)
retriever = get_self_query_retriever(
    vector_db.instance, llm=SELF_QUERY_MODEL, search_type="similarity", k=5
)
llm = get_llm(GENERATIVE_MODEL, os.getenv("OPENAI_API_KEY"))

prompt_template = get_prompt_template()
prompt_template_with_history = get_prompt_template_with_history()
