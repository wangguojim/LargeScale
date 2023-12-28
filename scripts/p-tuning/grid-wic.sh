
NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/mnt/yrfs/aohan/nodelist1"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

DATA_ROOT="/mnt/yrfs/aohan/data/superglue"

source $1 # model
source $2 # task

#CHECKPOINT_PATH="/mnt/yrfs/aohan/checkpoints/glm-130B-multitask-superglue-tp/global_step1500"
CHECKPOINT_PATH="/mnt/yrfs/aohan/checkpoints/glm-130B-tp/global_step49300"
MICRO_BATCH_SIZE=$(($BATCH_SIZE/$NUM_WORKERS))

for lr in 6e-3 8e-3; do
    for prefix in 256 128 64 32 16 8; do
        for bs in 8; do
            EXP_NAME=${TASK_NAME}-${NAME}-prefix-${prefix}-bsz-${bs}-lr-${lr}-${TIMESTAMP}

            LR_SINGLE=${lr}
            MICRO_BATCH_SIZE=$(($bs/$NUM_WORKERS))
            EPOCH_SINGLE=150

            mkdir -p logs-p-tuning/${EXP_NAME}

            TENSORBOARD_PATH="runs-p-tuning/${EXP_NAME}"

            args="./tasks/main.py \
                   --seed 1234 \
                   --task ${TASK_NAME} \
                   --pretrained-checkpoint ${CHECKPOINT_PATH} \
                   --train-data ${DATA_PATH} \
                   --micro-batch-size ${MICRO_BATCH_SIZE} \
                   --seq-length ${MAX_SEQ_LEN} \
                   --epochs ${EPOCH_SINGLE} \
                   --lr ${LR_SINGLE} \
                   --tensorboard-dir ${TENSORBOARD_PATH} \
                   --tensorboard-queue-size 5 \
                   --optimizer adam \
                   --weight-decay 1.0e-4 \
                   --tokenizer-type IceTokenizer \
                   --prefix-prompt-length ${prefix} \
                   --finetune \
                   --fp16 \
                   ${GLM_ARGS} \
                   ${TRAIN_ARGS} \
                   ${COMMON_ARGS} \
                    "

            run_cmd="PYTHONPATH=/thudm/LargeScale/packages ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${args}"
            echo ${run_cmd}
            eval ${run_cmd} 2>&1 | tee logs-p-tuning/${EXP_NAME}/output.log
        done
    done
done