#!/bin/bash
# docker build \
#     --pull \
#     --rm \
#     -f "Dockerfile" \
#     --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=hoge --build-arg PASSWORD=fuga \
#     -t \
#     patch_env:latest "."
docker build \
    --pull \
    --rm \
    -f "Dockerfile" \
    --build-arg UID=$(id -u) --build-arg GID=1000 --build-arg USER=hoge --build-arg PASSWORD=fuga \
    -t \
    patch_env:latest "."
