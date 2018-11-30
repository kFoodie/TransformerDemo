bin=../tensor2tensor/bin
python $bin/t2t-trainer --registry_help

export CUDA_VISIBLE_DEVICES=2

PROBLEM=translate_enzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu
HOME=`pwd`
DATA_DIR=$HOME/t2t_data
TMP_DIR=$DATA_DIR
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=2048' or even 1024.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --batch_size=4096 \
  --train_steps=100000 \
