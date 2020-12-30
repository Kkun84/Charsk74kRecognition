#!/bin/bash
docker exec -itd PatchSetsRL tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
