#!/bin/bash


OUTPUT_DIR="/your_path"
DATA_ROOT=="/your_path"
TRAIN_SET="your_train_set"

N_CLS=10
BATCH_SIZE=32
TRAIN_PROMPT_EPOCH=100
TRAIN_MODEL_EPOCH=100
CHECKPOINT_FREQ=10
PRINT_FREQ=300


python train.py \
  --output_dir ${OUTPUT_DIR} \
  --n_cls ${N_CLS} \
  --data_root ${DATA_ROOT} \
  --train_set ${TRAIN_SET} \
  --batch_size ${BATCH_SIZE} \
  --train_prompt_epoch ${TRAIN_PROMPT_EPOCH} \
  --train_model_epoch ${TRAIN_MODEL_EPOCH} \
  --print_freq ${PRINT_FREQ} \
  --checkpoint_freq ${CHECKPOINT_FREQ}
