import langchain
import openai
import dotenv, os, sys
from loguru import logger
from rich import inspect
from typing import Literal, Optional, Union
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from enum import Enum
from time import time

try: 
    from ..utils import check_gpu_availability
except:
    # local debug
    sys.path.append("/home/melion/Documents/Projects/AI-News-Digest/")
    from src.ai_news_digest.utils import check_gpu_availability

# TODO: 
# 1. add TEI (text-embedding-inference) or custom endpoint as an option
# 2. integrate this helper in the app code

EMBED_MODE = Literal["hf", "hf_infer_api", "hf_hub", "fastembed"]
EMBED_MODEL = Union[
    HuggingFaceEmbeddings, # in-python models
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceHubEmbeddings,  # API calls
    FastEmbedEmbeddings,
]

# define model kwargs
ENCODE_KWARGS = {
    "normalize_embeddings": True,
    "batch_size": 16,
    "output_value": "sentence_embedding",
    "convert_to_numpy": True,
    "show_progress_bar": True,
}


def load_embed_model(
    mode: EMBED_MODE="hf",
    model_name: str="BAAI/bge-small-en-v1.5",
    hf_hub_token: Optional[str]=None,
    hf_hub_api_token: Optional[str]=None,
    device: Optional[Literal["auto", "cpu", "cuda"]]=None,
) -> EMBED_MODEL:
    
    if mode=="hf":
        if device=="auto":
            device_ = check_gpu_availability()
        elif device in ["cpu", "cuda"]:
            device_ = device
        else:
            raise ValueError(f"device does not support `{device}` value")
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device_},
            encode_kwargs=ENCODE_KWARGS,
        )
    
    elif mode=="hf_infer_api":
        return HuggingFaceInferenceAPIEmbeddings(
            model_name=model_name,
            api_key=hf_hub_token,
        )
    
    elif mode=="hf_hub":
        return HuggingFaceHubEmbeddings(
            model=model_name,
            task="feature-extraction",
            huggingfacehub_api_token=hf_hub_api_token,
        )
    
    elif mode=="fastembed":
        return FastEmbedEmbeddings(
            model_name=model_name,
            max_length=512,
        )
    
    else:
        raise ValueError(f"`{mode}` mode is not supported")



if __name__=="__main__":

    # get config
    dotenv.load_dotenv()
    hf_token = os.environ["HUGGING_FACE_HUB_TOKEN"]
    model_name = os.environ["EMBED_MODEL_ID"]
    
    # init time df
    times = {}

    for mode in ["hf", "hf_infer_api", "hf_hub", "fastembed"]:
        
        # load model
        hf_embed_model = load_embed_model(
            mode=mode,
            model_name=model_name,
            device="auto" if mode=="hf" else None,
            hf_hub_token=hf_token if mode=="hf_infer_api" else None,
            hf_hub_api_token=hf_token if mode=="hf_hub" else None,
        )
        logger.success(f"Loaded model '{model_name}' successfully in '{mode}'")

        # run model & store time
        logger.info(f"Model running....")
        t0 = time()
        res = hf_embed_model.embed_query(text="Hello World !")
        times[mode] = time() - t0
        logger.success("Model run successful: ")
        print(res[:5])
    
    # display times
    print(times)