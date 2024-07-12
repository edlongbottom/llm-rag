from pathlib import Path

from langchain_core.documents.base import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Weaviate


def load_and_split_pdf(file_path: str) -> Document:

    pdf_loader = PyPDFLoader(file_path=file_path)
    pdf_chunks = pdf_loader.load()

    pdf_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        is_separator_regex=False,
    )
    pdf_splits = pdf_text_splitter.split_documents(pdf_chunks)
    return pdf_splits


def add_documents_to_store(
    weaviate_instance: Weaviate,
    documents: list[Document],
) -> list[str]:
    docs_added = weaviate_instance.add_documents(documents)
    return docs_added


def load_all_pdf_docs(weaviate_instance: Weaviate, data_dir: str) -> None:
    for file in Path(data_dir).glob("*.pdf"):

        pdf_splits = load_and_split_pdf(file)
        print(f"Split file {file.name} into {len(pdf_splits)} chunks.")

        docs_added = add_documents_to_store(weaviate_instance, pdf_splits)
        print(f"Added {len(docs_added)} chunks to vector db for file {file.name}.\n")
