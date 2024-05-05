# syntax=docker/dockerfile:1.7-labs

# nvidia-containers
# --> support/version matrix: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2023
# --> available images: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# --> test with: docker run --gpus all -it nvcr.io/nvidia/pytorch:23.06-py3 bash    
# FROM nvcr.io/nvidia/pytorch:23.06-py3

# fastapi image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# pytorch image
# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# install dependencies
RUN pip install --no-cache-dir TTS fastapi uvicorn
# RUN apt-get update
# RUN apt-get install git-lfs

# copy fastapi app module
RUN mkdir ./app
# COPY --parents ../../../src/ai_news_digest/steps/tts_utils.py ./app/
COPY ./src/ai_news_digest/steps/tts_utils.py ./app

# copy XTTS model
# --> assumes that the following command has been run in data/06_models/tts_api/
# --> $ git clone git@hf.co:coqui/XTTS-v2
# --> TODO: put that in the README or make a script
# COPY --parents ../../../data/06_models/tts_api/XTTS-v2/* ./data/06_models/tts_api/XTTS-v2
COPY ./data/06_models/tts_api/XTTS-v2/* ./data/06_models/tts_api/XTTS-v2
COPY ./data/06_models/tts_api/female_us_eng_johanna.mp3 ./data/06_models/tts_api/

# expose port
EXPOSE 8000

# launch fastapi server
CMD ["uvicorn", "app.tts_utils:app", "--host", "0.0.0.0", "--port", "8000"]

