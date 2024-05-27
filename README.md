# llm-rag
Repo for storing code for building a document chatbot using RAG and GPT


## Setup

### Configure your environment

Currently the Weaviate instance is configured to use OpenAI APIs for creating vector embeddings
from text, and to use GPT to generate model responses. 

Define the following environment variables in an `.env` file at the root of the repo:

```
OPENAI_API_KEY=""
HUGGING_FACE_API_KEY=""

WEAVIATE_URL="http://localhost:8080"
WEAVIATE_API_KEY=""
WEAVIATE_STORAGE=""
```

### Spin up the VectorDB

Docker is used to run a containerised instance of Weaviate. Use the following command to spin this up:

```bash
docker compose up -d
```