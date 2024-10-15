#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# Set parameters
SEED=2024
TEST_DATA=""


DATASET="chemprot"  #  chemprot ddi conll04 semeval2010
LEARNING_RATE=1e-5
NUM_EPOCHS=3


TRAIN_DATA=""

python relation_extraction.py \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --dataset $DATASET \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --model_save_path $MODEL_SAVE_PATH \
  --tokenizer_save_path $TOKENIZER_SAVE_PATH \
    --max_samples 3000 \
  --batch_size 16 \
    --seed $SEED




