set -x
DATA_DIR=$1
MODEL_DIR=$2
OUT_DIR=$3
ABLATION=$4
WEIGHTS=$5

mkdir -p "$OUT_DIR"

python -m bertrand.training.train_nopep \
  --input-dir=$DATA_DIR \
  --model-ckpt=$MODEL_DIR \
  --output-dir=$OUT_DIR \
  --n-splits=11 \
  --ablation=$ABLATION \
  --weights=$WEIGHTS

python -m bertrand.training.evaluate_nopep \
  --datasets-dir=$DATA_DIR \
  --results-dir=$OUT_DIR \
  --out=$OUT_DIR/results.csv \
  --ablation=$ABLATION \
  --weights=$WEIGHTS