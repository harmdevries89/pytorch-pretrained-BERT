#!/bin/bash

N_GPUS=$1
EXPERIMENT_DIR=$2
SQUAD_DIR=/home/hdvries/squad

DISTRIBUTED_ARGS="--nproc_per_node $N_GPUS"

export PYTHONPATH=$PWD:$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=8 \
 examples/run_squad.py \
 --bert_model bert-large-uncased \
 --do_train \
 --do_predict \
 --do_lower_case \
 --train_file $SQUAD_DIR/train-v1.1.json \
 --predict_file $SQUAD_DIR/dev-v1.1.json \
 --learning_rate 3e-5 \
 --num_train_epochs 2 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir $EXPERIMENT_DIR \
 --train_batch_size 24 \
 --gradient_accumulation_steps 12