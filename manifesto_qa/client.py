import os
import weaviate

from typing import Union


def get_weaviate_client() -> Union[weaviate.Client, bool]:

    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.auth.AuthApiKey(
            api_key=os.getenv("WEAVIATE_API_KEY")
        ),
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    )
    ready_flag = client.is_ready()
    return client, ready_flag


def get_batch_with_cursor(
    client: weaviate.Client,
    collection: str,
    batch_size: int,
    properties: list[str] = ["page", "source", "text"],
    cursor=None,
) -> list[dict]:
    query = (
        client.query.get(
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
    client: weaviate.Client,
    collection: str,
    batch_size: int = 20,
    properties: list[str] = ["page", "source", "text"],
) -> list[dict]:

    results = []
    cursor = None
    while True:

        next_batch = get_batch_with_cursor(
            client, collection, batch_size, properties, cursor=cursor
        )
        if len(next_batch) == 0:
            break

        results += next_batch
        cursor = next_batch[-1]["_additional"]["id"]

    return results
