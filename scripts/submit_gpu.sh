#! /bin/bash

NUM_WORKERS=96
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/thudm/LargeScale/wudao-130B"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

source $1
mkdir -p logs/${EXP_NAME}

run_cmd="PYTHONPATH=/thudm/LargeScale/packages ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${script_path} ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}/output.log
