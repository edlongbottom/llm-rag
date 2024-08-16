import os

from weaviate import Client
from weaviate.auth import AuthApiKey

from langchain.vectorstores import Weaviate


class VectorDB:
    def __init__(
        self,
        index_name: str,
        text_embeddings_model: str,
        generative_model: str,
    ) -> None:
        self.index_name = index_name
        self.text_embeddings_model = text_embeddings_model
        self.generative_model = generative_model

        self._client = self.init_client()

    @property
    def client(self) -> Client:
        return self._client

    @property
    def instance(self) -> Weaviate:
        return Weaviate(
            client=self._client,
            index_name=self.index_name,
            text_key="text",
            attributes=["source", "page", "text"],
        )

    def init_client(self) -> Client:
        return Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
            additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        )

    def reset_manifesto_schema(self) -> None:
        if self._client.schema.exists(self.index_name):
            self._client.schema.delete_class(self.index_name)

        self.create_manifesto_schema()

    def delete_all_schemas(self) -> None:
        self._client.schema.delete_all()

    def create_manifesto_schema(self) -> None:
        manifesto_qa_schema = {
            "class": self.index_name,
            "description": "Index storing political party GE 2024 manifesto documents.",
            "vectorizer": self.text_embeddings_model,
            "moduleConfig": {
                "generative-openai": {
                    "model": self.generative_model,
                }
            },
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The document content chunked",
                    "indexFilterable": True,
                    "indexSearchable": True,
                    "moduleConfig": {
                        self.text_embeddings_model: {
                            "vectorizePropertyName": True,
                            "tokenization": "lowercase",
                            "model": "text-embedding-3-small",
                            "dimensions": 1536,
                            "type": "text",
                        }
                    },
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "The source document (PDF)",
                    "indexFilterable": True,
                    "indexSearchable": True,
                    "moduleConfig": {
                        "text2vec-openai": {
                            "vectorizePropertyName": False,
                            "tokenization": "whitespace",
                        }
                    },
                },
                {
                    "name": "page",
                    "dataType": ["number"],
                    "description": "Page number",
                    "indexFilterable": True,
                    "indexSearchable": False,
                    "moduleConfig": {
                        "text2vec-openai": {
                            "vectorizePropertyName": False,
                            "tokenization": "whitespace",
                        }
                    },
                },
            ],
        }
        self._client.schema.create_class(manifesto_qa_schema)

    def get_batch_with_cursor(
        self,
        collection: str = "ManifestoQa",
        batch_size: int = 20,
        properties: list[str] = ["page", "source", "text"],
        cursor=None,
    ) -> list[dict]:
        query = (
            self.client.query.get(
                collection,
                properties,
            )
            .with_additional(["id"])
            .with_limit(batch_size)
        )

        if cursor is not None:
            result = query.with_after(cursor).do()
        else:
            result = query.do()

        return result["data"]["Get"][collection]

    def read_all_objects(
        self,
        collection: str = "ManifestoQa",
        batch_size: int = 20,
        properties: list[str] = ["page", "source", "text"],
    ) -> list[dict]:

        results = []
        cursor = None
        while True:

            next_batch = self.get_batch_with_cursor(
                collection, batch_size, properties, cursor=cursor
            )
            if len(next_batch) == 0:
                break

            results += next_batch
            cursor = next_batch[-1]["_additional"]["id"]

        return results
