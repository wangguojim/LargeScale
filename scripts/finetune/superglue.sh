#! /bin/bash

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/mnt/yrfs/aohan/nodelist1"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ibs11 NCCL_IB_HCA=mlx5_0,mlx5_1 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

DATA_ROOT="/mnt/yrfs/aohan/data/superglue"

source $1 # model
source $2 # task

TENSORBOARD_PATH="runs-finetune/${EXP_NAME}"
MICRO_BATCH_SIZE=$((BATCH_SIZE / 2 / (NUM_WORKERS * NUM_GPUS_PER_WORKER / TP_SIZE / PP_SIZE)))

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
#       --use-bnb-optimizer \

run_cmd="PYTHONPATH=/thudm/LargeScale/packages ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${args}"
echo ${run_cmd}
mkdir -p logs-finetune/${EXP_NAME}
eval ${run_cmd} 2>&1 | tee logs-finetune/${EXP_NAME}/output.log
