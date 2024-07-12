# llm-rag
Repo for storing code for building a document chatbot using RAG and GPT


## Next steps
- Trial different methods for vector search and document chunking to optimise the bot's performance (and for own understanding)
- Improve the chat interface by implementing memory (see - https://medium.com/snowflake/langchain-and-streamlit-rag-c5f53af8f6ba )
- Improve the chat by incorporating sources into the output
- Tidy up notebooks and other code
- Write up documentation


## Setup

### Configure your environment

Currently the Weaviate instance is configured to use OpenAI APIs for creating vector embeddings
from text, and to use GPT to generate model responses. 

Define the following environment variables in an `.env` file at the root of the repo:

```
OPENAI_API_KEY=""

WEAVIATE_URL="http://localhost:8080"
WEAVIATE_API_KEY=""
WEAVIATE_STORAGE=""
```

### Spin up the VectorDB

Docker is used to run a containerised instance of Weaviate. Use the following command to spin this up:

```bash
docker compose up -d
```

### Run the chatbot app

Streamlit is used to run the chatbot application. Run the following command at the terminal:

```bash
streamlit run manifesto_qa/chatbot.py
```

Navigate to `localhost:8501` in a browser to interact with the chatbot. 