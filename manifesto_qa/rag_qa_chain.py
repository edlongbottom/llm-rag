from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.base import BaseLanguageModel

from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQAWithSourcesChain


def get_prompt_template() -> ChatPromptTemplate:
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    return ChatPromptTemplate.from_template(template)


def get_rag_chain(llm: BaseLanguageModel, retriever: Weaviate) -> Runnable:
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | get_prompt_template()
        | llm
        | StrOutputParser()
    )


def get_rag_chain_with_sources(llm: BaseLanguageModel, retriever: Weaviate) -> Runnable:
    return RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever,
        question_prompt=get_prompt_template(),
    )


def ask_question(chain: Runnable, question: str) -> str:
    return chain.invoke(question)


def ask_question_with_sources(chain: Runnable, question: str) -> str:
    results = chain.invoke({"question": question}, return_only_outputs=True)
    return f"{results['answer']}\nSources: {results['sources']}"
