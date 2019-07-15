#!/bin/bash

MASTER=dgx01
NODES=(dgx01 dgx02 dgx04 dgx05 dgx06)
MASTER_IP=`dig +short ${MASTER}.elementai.net | tail -n 1`
NNODES=${#NODES[@]}
NGPUS=8
EXPERIMENT_DIR=$1

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Missing experiment dir argument"
    exit 1
fi

if [ -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: $EXPERIMENT_DIR already exists."
    exit 1
fi

mkdir -p ${EXPERIMENT_DIR}

for i in "${!NODES[@]}"; do
    HOST=${NODES[$i]}
    CMD="cd code/pytorch-pretrained-BERT && bash scripts/launch_docker_dist.sh \"bash scripts/run_pretraining_distributed.sh $MASTER_IP $NNODES $i $NGPUS $EXPERIMENT_DIR \""
    echo $CMD
    ssh $HOST $CMD &
done
