#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -p 5000:5000 \
    -p 6006:6006 \
    -p 8501:8501 \
    -p 8502:8502 \
    -p 8503:8503 \
    -p 8504:8504 \
    -p 8888:8888 \
    -it \
    --gpus=all \
    --ipc=host \
    --name=PatchEnv \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$DATASET:/dataset \
    patch_env:latest \
    ${@-fish}
