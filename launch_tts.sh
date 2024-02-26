docker run \
--rm \
-it \
-v ./data/07_model_output/tts-output:/root/tts-output \
-p 5002:5002 \
--gpus all \
--entrypoint /bin/bash \
ghcr.io/coqui-ai/tts


# python3 TTS/server/server.py --list_models
# python3 TTS/server/server.py --model_name tts_models/en/vctk/vits --use_cuda true



# docker run --rm --gpus all -v ~/tts-output:/root/tts-output ghcr.io/coqui-ai/tts --text "Hello." --out_path /root/tts-output/hello.wav --use_cuda true



# Server source code: 
#    --> https://github.com/coqui-ai/TTS/blob/dev/TTS/server/server.py

# Useful links:
#    --> python server TTS: https://github.com/coqui-ai/TTS/discussions/1634
#    --> cTTS (python TTS server) : https://github.com/thorstenMueller/cTTS