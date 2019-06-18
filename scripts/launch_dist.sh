#!/bin/bash

MASTER=dgx04
NODES=(dgx04 dgx05 dgx06)
MASTER_IP=`dig +short ${MASTER}.elementai.net | tail -n 1`
NNODES=${#NODES[@]}
NGPUS=8

echo $MASTER_IP

for i in "${!NODES[@]}"; do
    HOST=${NODES[$i]}
    CMD="cd pytorch-pretrained-BERT && bash scripts/launch_docker.sh \"bash scripts/run_pretraining_distributed.sh $MASTER_IP $NNODES $i $NGPUS\""
    echo $CMD
    ssh -tt $HOST $CMD &
done
