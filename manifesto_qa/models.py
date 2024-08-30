from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

from langchain_cohere import ChatCohere
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables.base import Runnable


def get_openai_llm(model_id: str, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(model=model_id, **kwargs)


def get_huggingface_llm(
    model_id: str, huggingface_api_token: str, **kwargs
) -> ChatHuggingFace:

    endpoint_url = f"https://api-inference.huggingface.co/models/{model_id}"
    llm = HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        huggingfacehub_api_token=huggingface_api_token,
        **kwargs,
    )
    return ChatHuggingFace(llm=llm, model_id=model_id)


def get_cohere_llm(model_id: str, **kwargs) -> ChatCohere:
    return ChatCohere(model=model_id, **kwargs)


def get_mistral_llm(model_id: str, **kwargs) -> ChatMistralAI:
    return ChatMistralAI(model_name=model_id, **kwargs)


def get_llm(model_id: str, **kwargs) -> Runnable:
    model_fns = {
        "gpt-3.5-turbo": get_openai_llm,
        "gpt-4o-mini": get_openai_llm,
        "gpt-4": get_openai_llm,
        "gpt-4-turbo": get_openai_llm,
        "deepset/roberta-base-squad2": get_huggingface_llm,
        "mistral-small": get_mistral_llm,
        "mistral-medium-latest": get_mistral_llm,
        "mistral-large-latest": get_mistral_llm,
        "command": get_cohere_llm,
        "command-light": get_cohere_llm,
        "command-r": get_cohere_llm,
    }
    model_fn = model_fns[model_id]
    return model_fn(model_id=model_id, **kwargs)
