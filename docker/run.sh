#!/bin/bash
# docker run \
#     -d \
#     --init \
#     --rm \
#     -p 8889:8888 \
#     -it \
#     --gpus=all \
#     --ipc=host \
#     --name=PatchEnv \
#     --env-file=.env \
#     --volume=$PWD:/workspace \
#     --volume=$DATASET:/dataset \
#     patch_env:latest \
#     ${@-fish}
docker run \
    -d \
    --init \
    --rm \
    -it \
    --ipc=host \
    --name=PatchEnv \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$DATASET:/dataset \
    patch_env:latest \
    ${@-fish}
