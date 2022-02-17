#!/usr/bin/env bash

set -e

: ${DATA_DIR:=/data2/tuong/OLLI-SPEECH-1.8/LJSpeech_style}
: ${ARGS="--extract-mels"}

python3.8 prepare_dataset.py \
    --wav-text-filelists filelists/filelist_combine.txt\
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    --n-speakers 2 \
    $ARGS
