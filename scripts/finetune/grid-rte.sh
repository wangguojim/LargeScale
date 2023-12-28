#! /bin/bash

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/mnt/yrfs/aohan/nodelist2"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ibs11 NCCL_IB_HCA=mlx5_0,mlx5_1 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

DATA_ROOT="/mnt/yrfs/aohan/data/superglue"

source $1 # model
source config_tasks/task_rte.sh # task

for bs in 16 32; do
    for lr in 5e-6 1e-5 2e-5; do
        EXP_NAME=${TASK_NAME}-${NAME}-bsz-${bs}-lr-${lr}-${TIMESTAMP}
        TENSORBOARD_PATH="runs-finetune/${EXP_NAME}"
        MICRO_BATCH_SIZE=$(($bs/(NUM_WORKERS * NUM_GPUS_PER_WORKER / TP_SIZE / PP_SIZE)))
        LR=${lr}

        args="./tasks/main.py \
               --seed 1234 \
               --task ${TASK_NAME} \
               --pretrained-checkpoint ${CHECKPOINT_PATH} \
               --train-data ${DATA_PATH} \
               --tensorboard-dir ${TENSORBOARD_PATH} \
               --micro-batch-size ${MICRO_BATCH_SIZE} \
               --global-batch-size ${BATCH_SIZE} \
               --seq-length ${MAX_SEQ_LEN} \
               --epochs ${EPOCH} \
               --lr ${LR} \
               --tensorboard-queue-size 5 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --adam-eps 1e-8 \
               --weight-decay 1.0e-1 \
               --initial-loss-scale 65536 \
               --tokenizer-type IceTokenizer \
               --checkpoint-activations \
               --freeze-prefix-layer-num 20 \
               --finetune \
               --fp16 \
               ${GLM_ARGS} \
               ${TRAIN_ARGS} \
               ${COMMON_ARGS} \
                "

        run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${args}"
        echo ${run_cmd}
        mkdir -p logs-finetune/${EXP_NAME}
        eval ${run_cmd} 2>&1 | tee logs-finetune/${EXP_NAME}/output.log
    done
done
