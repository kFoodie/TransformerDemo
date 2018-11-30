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

# Decode
t2t-decoder --data_dir=./t2t_data
    --problem=translate_enzh_wmt32k
    --model=transformer
    --hparams_set=transformer_base_single_gpu
    --output_dir=./t2t_train/translate_enzh_wmt32k/transformer-transformer_base_single_gpu/
    --decode_hparams="beam_size=4,alpha=0.6"
    --decode_from_file=./t2t_data/wmt_enzh_32768k_tok_dev.lang1
    --decode_to_file=translation.zh

t2t-bleu --translation=./t2t_data/translation.zh
    --reference=./t2t_data/wmt_enzh_32768k_tok_dev.lang2
