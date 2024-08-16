from langchain.llms import OpenAI
from langchain.vectorstores import Weaviate

from langchain.retrievers.self_query.base import SelfQueryRetriever
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
