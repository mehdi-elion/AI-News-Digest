# choose model 
# model=meta-llama/Llama-2-7b-chat-hf
model=C:/Users/mehdi/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf

# share a volume with the Docker container to avoid downloading weights every run
volume=$PWD/data/06_models/gen_api

# get Hf_hub token
token=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

# run docker command
docker run \
    --gpus all \
    --shm-size 1g \
    -e HUGGING_FACE_HUB_TOKEN=$token \
    -p 8080:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:1.3 \
    --model-id $model \
    --dtype float16 \
    # --quantize bitsandbytes \
    