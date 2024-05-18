from TTS.api import TTS
import torch
from typing import Union, Dict, Any, Tuple, Optional, Literal
import numpy as np
from numpy.typing import ArrayLike
import json
from loguru import logger
import requests
import os
import dotenv
import runpod





# func to request local TTS FastAPI Endpoints (XTTS-v2: anghelion/xtts_fastapi)
def request_tts_fastapi(
    text: str,
    tts_url: str="http://127.0.0.1:8000",
    endpoint: str="tts2",
) -> Optional[Tuple[str, ArrayLike, int, int]]:
    
    # build full url (defaults to : http://127.0.0.1:8000/tts2/)
    url = f"{tts_url}/{endpoint}/"

    # send get request
    tts_params = {"text": text}
    r = requests.get(url=url, params=tts_params)

    # parse results
    r_text = r.json()["text"]
    wav = r.json()["wav"]
    output_sample_rate = r.json()["output_sample_rate"]
    sample_rate = r.json()["sample_rate"]

    # return results
    return r_text, wav, output_sample_rate, sample_rate


# func to request remote TTS (XTTS-v2: anghelion/xtts_runpod)
def request_tts_runpod(
    text: str,
    endpoint_id: str,
    api_key: str,
    timeout: int=60,
) -> Optional[Tuple[str, ArrayLike, int, int]]:
    # set credentials
    runpod.api_key = api_key

    # build input payload
    input_payload = {"input": {"text": text}}

    # send request
    try:

        # endpoint request
        endpoint = runpod.Endpoint(endpoint_id)
        run_request = endpoint.run(input_payload)

        # initial check without blocking, useful for quick tasks
        status = run_request.status()
        logger.info(f"Initial Runpod job status: {status}")

        if status != "COMPLETED":
            # Polling with timeout for long-running tasks
            output = run_request.output(timeout=timeout)
        else:
            output = run_request.output()
        logger.info(f"Runpod Job status: {run_request.status()}")

        # parse and send results
        r_text = output["text"]
        wav = output["wav"]
        output_sample_rate = output["output_sample_rate"]
        sample_rate = output["sample_rate"]
        return r_text, wav, output_sample_rate, sample_rate 

    except Exception as e:
        logger.error(f"Runpod status: An error occurred: {e}")
        return None



# func to request an in-memory (i.e. in-python) TTS model using Coqui
def run_tts_model(
    text: str,
    model: Optional[TTS]=None,
    model_path: str="../data/06_models/tts_api/XTTS-v2",
    config_path: str="../data/06_models/tts_api/XTTS-v2/config.json",
    speaker_wav_path: str="../data/06_models/tts_api/female_us_eng_johanna.mp3",
    device: Literal["cpu", "cuda", "auto"]="auto",
) -> Tuple[str, ArrayLike, int, int]:
    
    # set device
    if device in ["cpu", "cuda"]:
        device_ = device
    elif device == "auto":
        device_ = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        raise ValueError(f"`device` argument does not support value `{device}`")
    logger.info(f"device set to {device_}")

    # load model
    if model is None:
        tts = TTS(model_path=model_path, config_path=config_path)
    else:
        tts = model
    
    # send it to chosen device
    tts = tts.to(device_)

    # run inference
    wav = tts.tts(text=text, speaker_wav=speaker_wav_path, language="en")

    # send results
    return (
        text, 
        wav, 
        tts.config.audio.output_sample_rate, 
        tts.config.audio.sample_rate 
    )
