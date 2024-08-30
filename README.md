# UK General Election Manifesto Q&A with RAG

Code for building a chatbot that answers questions about UK political party policies as
written in their manifestos for the 2024 general election, using Retrieval Augmented
Generation (RAG).

<img src="images/chat_example.png" alt="drawing" width="700"/>


Technologies include:
- Streamlit for creating and running the chat interface
- Weaviate as the vector database for storage and retrieval of vector embeddings of manifesto content
- Integrations with LLMs hosted by OpenAI, Cohere, Mistral and HuggingFace for question answering
- LangChain for chaining together document loading/chunking, context retrieval and generation
- Docker compose is used to host the Weaviate instance



## Next steps
- Explore how the bot's performance changes with different ML models
    - Test out the app with a few different generative models
    - Consider how you would make the system flexible for different LLMs for vector embeddings

Other improvements to consider:
- Different vector search methods (e.g. hybrid, keyword, BM25)
- Different document loading/chunking approaches
- Use compression to reduce the size of the context being passed (save costs)
- Considerations for deployment of the chatbot


## Setup

### Configure your environment

The application supports using generative large language models from various sources (OpenAI, Cohere, Mistral 
and HuggingFace) for question answering. The vector database is currently configured to use an OpenAI model for 
creating and retrieving vector embeddings. API keys must first be created to use the different model APIs.

Provide a local path for `WEAVIATE_STORAGE` so that the vector database data is persisted. 

`DATA_DIR` defines the location of the documents (party manifesto PDF files)

Populate the following environment variables as required in an `.env` file at the root of the repo:

```
OPENAI_API_KEY=""
HUGGINGFACEHUB_API_TOKEN=""
COHERE_API_KEY=""
MISTRAL_API_KEY=""

WEAVIATE_URL="http://localhost:8080"
WEAVIATE_API_KEY=""
WEAVIATE_STORAGE=""
DATA_DIR = ""
```

### Spin up the VectorDB

Docker is used to run a containerised instance of Weaviate. Use the following command to spin this up:

```bash
docker compose up -d
```

### Install the application dependencies

This project uses poetry for managing python dependencies. Install poetry and then run the folllowing: 

```bash
poetry install
```

### Run the chatbot app

Streamlit is used to run the chatbot application. Run the following command at the terminal:

```bash
streamlit run manifesto_qa/app.py
```

Navigate to `localhost:8501` in a browser to interact with the chatbot. 