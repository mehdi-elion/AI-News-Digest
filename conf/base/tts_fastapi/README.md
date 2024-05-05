# TTS FastAPI Endpoint

## How to build it
From the root directory (not this `conf/base/tts_fastapi/` directory), run the following command:
```bash
docker build -f conf/base/tts_fastapi/Dockerfile . --tag xtts_fastapi:latest
```

## How to run it
```bash
docker run --gpus all --shm-size=4gb -p 8000:8000 xtts_fastapi:latest 
```

or

```bash
docker run --gpus all --shm-size=4gb -p 8000:8000 anghelion/xtts_fastapi:latest 
```


To run it in interactive mode:
```bash
docker run -it --gpus all --shm-size=4gb -p 8000:8000 xtts_fastapi:latest bash 
```

or

```bash
docker run -it --gpus all --shm-size=4gb -p 8000:8000 anghelion/xtts_fastapi:latest bash 
```

## Push the image to Dockerhub
```bash
docker push xtts_fastapi:latest
docker push anghelion/xtts_fastapi:latest
```
