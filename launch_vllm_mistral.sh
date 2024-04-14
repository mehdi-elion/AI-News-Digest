# choose model
model=mistralai/Mistral-7B-Instruct-v0.2

# share a volume with the Docker container to avoid downloading weights every run
# volume=$PWD/data/06_models/vllm_api
hf_cache=/mnt/c/Users/mehdi/.cache/huggingface/hub
echo $volume:/data

# export & get Hf_hub token
export $(cat conf/local/.env | xargs)
HF_TOKEN=$HF_token

# login to hf hub
huggingface-cli login --token $HF_TOKEN

# run docker command
docker run \
    --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -p 8000:8000 \
    -v $hf_cache:/root/.cache/huggingface/hub \
    ghcr.io/mistralai/mistral-src/vllm:latest \
    --host 0.0.0.0 \
    --model $model
    # --model $hf_hub_cache
    # -v $volume:/data \
