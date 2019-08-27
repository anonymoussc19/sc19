#!/bin/bash

sbatch -t 1200 -p batch-bdw-k80 ./run.sh> batch.out