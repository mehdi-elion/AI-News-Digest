# choose model
model=meta-llama/Llama-2-7b-chat-hf
hf_hub_cache=/mnt/c/Users/mehdi/.cache/huggingface/hub
# local_model=/data/models--meta-llama--Llama-2-7b-chat-hf

# share a volume with the Docker container to avoid downloading weights every run
volume=$PWD/data/06_models/gen_api
echo $volume:/data

# export & get Hf_hub token
export $(cat conf/local/.env | xargs)
token=$HF_token

# login to hf hub
huggingface-cli login --token $token

# run docker command
docker run \
    --gpus all \
    --shm-size 20g \
    -e HUGGING_FACE_HUB_TOKEN=$token \
    -p 8080:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:1.3 \
    --model-id $model \
    --quantize eetq \
    # --dtype float16 \
    # --huggingface-hub-cache hf_hub_cache \
