

#! /bin/bash

DATA_PATH=""
MULTITASK_DATA_PATH=""
NAME="GLM-130B"

EXP_NAME=${NAME}-${TIMESTAMP}
CHECKPOINT_PATH="/thudm/LargeScale/checkpoints/${NAME}"
TENSORBOARD_PATH="runs/glm-130b/${NAME}"

config_json="./ds-configs/${EXP_NAME}/ds_config.json"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4224 # 176 * 24

TP_SIZE=4
PP_SIZE=8

NHIDDEN=12288
FFN_HIDDEN=$((NHIDDEN * 8 / 3))
NLAYERS=70
NHEADS=96
LENGTH_PER_SAMPLE=2000 # sequence length per sample from BinaryDataset
SEQ_LEN=2048 # actual length during training (pad to this)

SAVE_INTERVAL=100

TRAIN_TOKENS=450000000000 # 450B tokens
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 5 / 1000))  # 0.5% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup

ZERO_STAGE=1

script_path="pretrain_glm.py"

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 4e-5 \
    --min-lr 4e-6 \
    --override-lr-scheduler \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 3 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

GLM_ARGS="
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NHEADS \
       --make-vocab-size-divisible-by 768 \
       --glm \
       --gpt-prob 0.7 \
       --single-span-prob 0.02 \
       --short-seq-prob 0.02 \
       --mask-prob 0.15 \
       --average-block-length 3 \
       --min-gmask-ratio 0.2 \
       --aggregated-samples-per-sequence 4 \
       --deepnorm \
       --position-embedding-type rotary \
       --ffn-hidden-size $FFN_HIDDEN \
       --glu-activation geglu \
       --no-bias-gelu-fusion \
    "

DEEPSPEED_ARGS=" \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --zero-stage $ZERO_STAGE \
       --partition-activations \
       --deepspeed-activation-checkpointing \
    "

gpt_options=" \
       $GLM_ARGS \
       --pp-partition-method 'type:transformer|embedding' \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --rampup-batch-size 192 24 $BATCH_WARMUP_SAMPLES \
       --train-samples $TRAIN_SAMPLES \
       --length-per-sample $LENGTH_PER_SAMPLE \
       --seq-length $SEQ_LEN \
       --multitask-data-path $MULTITASK_DATA_PATH \
       --multitask-ratio 0.05 \
       --num-workers 1 \
       --data-path $DATA_PATH \
       --skip-train-iteration-range 40701-40900 42401-42600 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --abort-on-unmet-fused-kernel-constraints \
       --split 949,50,1 \
       --distributed-backend nccl \
       --checkpoint-activations \
       --init-method-std 0.0052 \
       --shrink-logit-embedding-gradient \
       --shrink-embedding-gradient-alpha 0.1 \
       --fp16 \
       $OPTIMIZER_ARGS \
       $DEEPSPEED_ARGS \
       $OUTPUT_ARGS
"

mkdir -p ds-configs/${EXP_NAME}
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 200,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "steps_per_print": 1,
  "wall_clock_breakdown": true
}
EOT
