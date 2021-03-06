#!/bin/bash
docker build \
    --pull \
    --rm \
    -f "Dockerfile" \
    --build-arg UID=$(id -u) --build-arg USER=hoge --build-arg PASSWORD=fuga \
    -t \
    charsk74k_recognition:latest "."
