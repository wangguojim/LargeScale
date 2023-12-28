TASK_NAME=WSC
EXP_NAME=wsc-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/WSC-negative"
MAX_SEQ_LEN=128

LR=1e-5
EPOCH=50

LR_PT=5e-3
EPOCH_PT=150

TRAIN_ARGS="--lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --wsc-negative \
            --length-penalty 1 \
            --loss-func mix \
            --fast-decode \
            --pattern-id 3"

COMMON_ARGS="--save-interval 10000000 \
             --log-interval 1 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16