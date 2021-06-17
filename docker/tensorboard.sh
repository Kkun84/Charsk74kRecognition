#!/bin/bash
docker exec -itd charsk74k_recognition tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
