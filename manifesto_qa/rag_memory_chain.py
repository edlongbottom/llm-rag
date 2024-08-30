from typing import Union

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.base import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.memory import ChatMessageHistory


def get_history_aware_retriever(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> BaseRetriever:

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def get_qa_chain(llm: BaseLanguageModel) -> Runnable:

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)


def get_rag_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:

    history_aware_retriever = get_history_aware_retriever(llm, retriever)
    qa_chain = get_qa_chain(llm)

    return create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=qa_chain
    )


def get_rag_chain_with_memory(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    chat_memory: ChatMessageHistory = ChatMessageHistory(),
) -> RunnableWithMessageHistory:

    rag_chain = get_rag_chain(llm, retriever)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def ask_question_with_history(
    chain: Runnable, question: str, debug=False
) -> Union[dict, str]:
    response = chain.invoke(
        {"input": question}, config={"configurable": {"session_id": "abc"}}
    )
    if debug:
        return response
    else:
        return response["answer"]
