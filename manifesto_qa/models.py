from langchain_community.chat_models import ChatOpenAI


def get_llm(openai_model: str, openai_api_key: str, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(model=openai_model, api_key=openai_api_key, **kwargs)
