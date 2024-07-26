from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import Weaviate

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.query_constructor.base import AttributeInfo


def get_retriever(weaviate_instance: Weaviate, **kwargs):
    return weaviate_instance.as_retriever(**kwargs)


def get_self_query_retriever(
    weaviate_instance: Weaviate, llm: str, **kwargs
) -> SelfQueryRetriever:
    return SelfQueryRetriever.from_llm(
        llm=OpenAI(model=llm, temperature=0),
        vectorstore=weaviate_instance,
        document_contents="Manifestos",
        metadata_field_info=[
            AttributeInfo(
                name="source",
                description="The manifesto PDF the chunk is from",
                type="string",
            ),
            AttributeInfo(
                name="page",
                description="The page from the manifesto",
                type="integer",
            ),
        ],
        verbose=True,
        **kwargs,
    )


def get_llm(openai_model: str, openai_api_key: str, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(model=openai_model, api_key=openai_api_key, **kwargs)


def get_prompt_template() -> ChatPromptTemplate:
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    return ChatPromptTemplate.from_template(template)


def get_prompt_template_with_history() -> ChatPromptTemplate:
    # TODO: incomplete
    system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Context: {context}
            Question: """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


def run_rag_chain(
    retriever: Weaviate,
    llm: ChatOpenAI,
    prompt_template: ChatPromptTemplate,
    prompt: str,
) -> str:
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    output = rag_chain.invoke(prompt)
    return output


def run_rag_chain_with_sources(
    retriever: Weaviate,
    llm: ChatOpenAI,
    prompt_template: ChatPromptTemplate,
    prompt: str,
) -> str:
    rag_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever,
        question_prompt=prompt_template,
    )
    results = rag_chain.invoke({"question": prompt}, return_only_outputs=True)
    return f"{results['answer']}\nSources: {results['sources']}"


def run_rag_chain_with_history(
    retriever: Weaviate,
    llm: ChatOpenAI,
    prompt_template: ChatPromptTemplate,
    msgs: StreamlitChatMessageHistory,
    prompt: str,
) -> str:
    # TODO: incomplete
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="history",
    )
    config = {"configurable": {"session_id": "any"}}
    return chain_with_history.invoke({"question": prompt}, config)
