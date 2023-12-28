TASK_NAME=wsc
EXP_NAME=wsc_generative-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/WSC"
MAX_SEQ_LEN=128

LR_SINGLE=1e-5
EPOCH_SINGLE=50
XXLARGE_EPOCH=100

TRAIN_ARGS="--lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1e-4"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

BATCH_SIZE=16
