EXP_NAME=record-${NAME}-${TIMESTAMP}
TASK_NAME=ReCoRD
DATA_PATH="${DATA_ROOT}/ReCoRD"
MAX_SEQ_LEN=512

LR=1e-5
EPOCH=3

LR_PT=5e-3
EPOCH_PT=12

TRAIN_ARGS="--lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 1000000 \
             --log-interval 1 \
             --eval-interval 1000000 \
             --eval-iters 100 \
             --fast-decode \
             --tgt-seq-length 32"


PATTERN_IDS=(0)

BATCH_SIZE=32