#!/bin/bash
docker exec -itd PatchEnv tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
