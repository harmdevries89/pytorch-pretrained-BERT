#!/bin/bash

# Change for multinode config
MASTER_ADDR=$1
MASTER_PORT=1234
NNODES=$2
NODE_RANK=$3
N_GPUS=$4
EXPERIMENT_DIR=$5

DISTRIBUTED_ARGS="--nproc_per_node $N_GPUS --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use_env"

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_JIT=0

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  examples/run_pretraining_bert.py \
    --data-path /home/hdvries/data/prep_128.txt \
    --exp-dir ${EXPERIMENT_DIR} \
    --load-ckpt /home/hdvries/exp/test_args/checkpoints/chkpt.500.pt \
    --max-position-embeddings 512 \
    --max-seq-length 128 \
    --num-layers 24 \
    --hidden-size 1024 \
    --intermediate-size 4096 \
    --num-attention-heads 16 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --train-iters 1000000 \
    --report-every 100 \
    --save-every 500 \
    --num-workers 1 \
    --learning-rate 1e-4 \
    --warmup-proportion 0.01 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --batch-size 20 \
    --use-fp16 \
    --layernorm-fp32 \
    --checkpoint-act-fn \
    > ${EXPERIMENT_DIR}/node.${NODE_RANK}.log 2> ${EXPERIMENT_DIR}/node.${NODE_RANK}.err.log
