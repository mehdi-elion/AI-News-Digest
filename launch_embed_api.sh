# choose model 
model=BAAI/bge-small-en

# choose revision
revision=main

# share a volume with the Docker container to avoid downloading weights every run
volume=$PWD/data/06_models/embed_api

# choose device: # 'docker run --gpus $device' 
# device='"device=0"'
device=all

# run docker command
docker run -p 8080:80 \
    -v $volume:/data \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:0.6 \
    --model-id $model \
    --revision $revision