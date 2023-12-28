#! /bin/bash

WORLD_SIZE=8
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

source $1 # model

CHECKPOINT_PATH="/thudm/LargeScale/checkpoints/wudao-130B-megatron-8tp/global_step20000"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./glm/generate.py \
       --seed 1234 \
       --tokenizer-type IceTokenizer \
       --temperature 1.0 \
       --load ${CHECKPOINT_PATH} \
       --genfile sample_input_file.out \
       --top_p 0.9 \
       --micro-batch-size 1 \
       --seq-length 2048 \
       --out-seq-length 2048 \
       --log-interval 1 \
       --num-samples 4 \
       --benchmark \
       --fp16 \
       ${GLM_ARGS}
