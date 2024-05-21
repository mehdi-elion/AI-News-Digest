# TODO list
# 1. [DONE] helper to interact with groqcloud
# 2. [DONE] helper to interact with runpod / vLLM
# 3. [DONE] helper to interact with OpenRouter
#  --> c.f. https://medium.com/@gal.peretz/openrouter-langchain-leverage-opensource-models-without-the-ops-hassle-9ffbf0016da7
# 4. helper to interact with huggingchat API
# 5. Inegrate this module into the app script

import os, dotenv
from langchain_community.llms import VLLMOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from typing import Optional
from loguru import logger
from time import time

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)


def load_openrouter(
    openrouter_api_key: str,
    model_name: str="mistralai/mistral-7b-instruct:free",
    openai_api_base: str="https://openrouter.ai/api/v1",
    **kwargs: dict,
) -> ChatOpenRouter:
    """Set connector to Openrouter LLM Inference API.

    Parameters
    ----------
    openrouter_api_key : str
        Openrouter API token.
    model_name : str, optional
        Name of the model, by default "mistralai/mistral-7b-instruct:free". 
        Available models can be browsed at https://openrouter.ai/docs#models
    openai_api_base : str, optional
        Openrouter endpoint, by default "https://openrouter.ai/api/v1"

    Returns
    -------
    ChatOpenRouter
        Langchain-compatible llm (connector) object
    """
    
    return ChatOpenRouter(
        model_name=model_name,
        openai_api_key=openrouter_api_key,
        openai_api_base=openai_api_base,
        **kwargs,
    )
  

def load_vllm(
    openai_api_key: str,
    openai_api_base: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    **kwargs: dict,
) -> VLLMOpenAI:
    """Wrap the OpenAI-compatible vLLM API for streamlit caching.

    Parameters
    ----------
    openai_api_key : str
        OpenAI API key, passed to `VLLMOpenAI`. Can be set to "EMPTY" when using 
        vLLM locally.
    openai_api_base : _type_, optional
        OpenAI API url, passed to `VLLMOpenAI`, Can be set to
        "http://localhost:8000/v1" when using vLLM locally. 
    model_name : str
        Name of the model being served, by default 
        "mistralai/Mistral-7B-Instruct-v0.2"

    Returns
    -------
    llm : VLLMOpenAI
        Connector to the LLM being served through the vLLM API

    """
    llm = VLLMOpenAI(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        model_name=model_name,
        **kwargs,
    )
    return llm


def load_groq(
    groq_api_key: str="YOUR_API_KEY", 
    model_name: str="mixtral-8x7b-32768",
    **kwargs: dict,
) -> ChatGroq:
    
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model_name,
        **kwargs,
    )


if __name__=="__main__":

    dotenv.load_dotenv()

    # initiate times dict
    times = {}

    # Runpod - vLLM
    logger.info("Using vLLM-Runpod...")
    llm = load_vllm(
        openai_api_key=os.environ["RUNPOD_API_KEY"],
        openai_api_base=os.environ["RUNPOD_VLLM_ENDPOINT"],
    )
    t0 = time()
    res = llm.predict("Hello, how are you ?")
    times["Runpod-vLLM"] = time() - t0
    print("---- anwser ----: \n", res)
    
    
    # OpenRouter
    logger.info("Using OpenRouter...")
    llm = load_openrouter(openrouter_api_key=os.environ["OPENROUTER_API_KEY"])
    t0 = time()
    res = llm.predict("Hello, how are you ?")
    times["OpenRouter"] = time() - t0
    print("---- anwser ----: \n", res)

    # GroqCloud
    logger.info("Using GroqCloud...")
    llm = load_groq(groq_api_key=os.environ["GROQ_CLOUD_API_KEY"])
    t0 = time()
    res = llm.predict("Hello, how are you ?")
    times["GroqCloud"] = time() - t0
    print("---- anwser ----: \n", res)

    # display times
    logger.info(f"Execution times: \n{times}")