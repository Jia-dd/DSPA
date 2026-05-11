#!/bin/bash


N_CLS=10

DATA_ROOT=="/your_path"

BATCH_SIZE=256

WEIGHTS_PATH="/weights_path"

TEST_SET="your_test_set"


python predict.py \
  --n_cls ${N_CLS} \
  --data_root ${DATA_ROOT} \
  --batch_size ${BATCH_SIZE} \
  --weights_path ${WEIGHTS_PATH} \
  --test_set ${TEST_SET}
