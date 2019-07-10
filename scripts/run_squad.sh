#!/bin/bash

N_GPUS=$1
EXPERIMENT_DIR=$2


DISTRIBUTED_ARGS="--nproc_per_node $N_GPUS"

export PYTHONPATH=$PWD:$PYTHONPATH

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  examples/run_squad.py \
  --do_train \
  --do_predict \
  --bert_model bert-large-cased \
  --train_file /home/hdvries/data/train-v1.1.json \
  --predict_file /home/hdvries/data/dev-v1.1.json \
  --train_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir $EXPERIMENT_DIR \