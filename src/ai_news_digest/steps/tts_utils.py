from TTS.api import TTS
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Union, Dict, Any
import numpy as np
import json



# if __name__=="__main__":
        
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device = ", device)

tts = TTS(
    # model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    # model_name="data/06_models/tts_api/XTTS-v2",
    # model_name="",
    model_path="data/06_models/tts_api/XTTS-v2",
    config_path="data/06_models/tts_api/XTTS-v2/config.json"
).to(device)

print("Loaded model tts !!")

# generate speech by cloning a voice using default settings
# tts.tts_to_file(
#     text="Elon musk claims he bought the last Open AI shares from Microsoft.",
#     file_path="data/07_model_output/tts-output/output.wav",
#     speaker_wav="data/06_models/tts_api/female_us_eng_johanna.mp3",
#     language="en"
# )


# wav = tts.tts(
#     text="Elon musk claims he bought the last Open AI shares from Microsoft.",
#     speaker_wav="data/06_models/tts_api/female_us_eng_johanna.mp3",
#     # speaker="241",
#     language="en",
# )

# print(wav)
# print(tts.config.audio)
# print(tts.config.audio.output_sample_rate)
# print(tts.config.audio.sample_rate)


# print("Model loaded !!!")


#-------- APP ----------
# $ uvicorn src.ai_news_digest.steps.tts_utils:app --reload

app = FastAPI()


@app.get("/")
def get_default_model():
    return {"model_name": "World"}


@app.get("/tts/")
def read_text(text: str):

    wav = tts.tts(
        text=text,
        # speaker_wav="data/06_models/tts_api/female_us_eng_johanna.mp3",
        speaker_wav="data/06_models/tts_api/XTTS-v2/samples/en_sample.wav",
        language="en",
    )
    
    print("ran inference successfully !!!!!!")
    print(type(wav))

    json_data = jsonable_encoder({
        "text": text, 
        "wav": [float(x) for x in wav],  # FastAPI does not support numpy types, only support native python types
        "output_sample_rate": tts.config.audio.output_sample_rate,
        "sample_rate": tts.config.audio.sample_rate,
    })

    return JSONResponse(content=json_data)


@app.get("/tts2/")
def read_text2(text: str):

    wav = tts.tts(
        text=text,
        speaker_wav="data/06_models/tts_api/female_us_eng_johanna.mp3",
        # speaker_wav="data/06_models/tts_api/XTTS-v2/en_sample.wav",
        language="en",
    )
    
    print("ran inference successfully !!!!!!")
    print(type(wav))

    return {
        "text": text, 
        "wav": [float(x) for x in wav],  # FastAPI does not support numpy types, only support native python types
        "output_sample_rate": tts.config.audio.output_sample_rate,
        "sample_rate": tts.config.audio.sample_rate,
    }
