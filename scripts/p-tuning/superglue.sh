#! /bin/bash

NUM_WORKERS=8
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/mnt/yrfs/aohan/nodelist2"
#HOST_FILE_PATH="/thudm/LargeScale/wudao-1"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

DATA_ROOT="/mnt/yrfs/aohan/data/superglue"
#DATA_ROOT="/thudm/LargeScale/data/superglue"

source $1 # model
source $2 # task

mkdir -p logs-p-tuning/${EXP_NAME}

MICRO_BATCH_SIZE=4

TENSORBOARD_PATH="runs-p-tuning/${EXP_NAME}"

args="./tasks/main.py \
       --seed 1234 \
       --task ${TASK_NAME} \
       --pretrained-checkpoint ${CHECKPOINT_PATH} \
       --train-data ${DATA_PATH} \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --global-batch-size ${BATCH_SIZE} \
       --seq-length ${MAX_SEQ_LEN} \
       --epochs ${EPOCH_PT} \
       --lr ${LR_PT} \
       --tensorboard-dir ${TENSORBOARD_PATH} \
       --tensorboard-queue-size 5 \
       --optimizer adam \
       --weight-decay 1.0e-4 \
       --tokenizer-type IceTokenizer \
       --prefix-prompt-length 16 \
       --finetune \
       --fp16 \
       ${GLM_ARGS} \
       ${TRAIN_ARGS} \
       ${COMMON_ARGS} \
        "

run_cmd="PYTHONPATH=/thudm/LargeScale/packages ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${args}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee logs-p-tuning/${EXP_NAME}/output.log
