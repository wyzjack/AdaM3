#!/bin/bash


DATA_PATH=./data-bin/iwslt14.tokenized.de-en.joined
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2

NUM=5

# seed 0 

OUTPUT_PATH=log/sgd_0
mkdir -p $OUTPUT_PATH
export CUDA_VISIBLE_DEVICES=0; python main.py ${DATA_PATH} \
                            --seed 0 \
                            --momentum 0.9 \
                            --arch ${ARCH} --share-all-embeddings \
                            --optimizer sgd  --clip-norm 0.0 \
                            --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                            --criterion label_smoothed_cross_entropy \
                            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
                            --lr 0.1 --min-lr 1e-9 \
                            --label-smoothing 0.1 --weight-decay 0.0001 \
                            --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                            --update-freq 1 --no-progress-bar --log-interval 50 \
                            --ddp-backend no_c10d \
                            --keep-last-epochs ${NUM} --max-epoch 55 \
                            --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                            | tee -a ${OUTPUT_PATH}/train_log.txt

--early-stop ${NUM} \

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

PYTHONIOENCODING=utf-8 python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
--target-lang en \
> ${RESULT_PATH}/res.txt

# seed 1

OUTPUT_PATH=log/sgd_1
mkdir -p $OUTPUT_PATH
export CUDA_VISIBLE_DEVICES=0; python main.py ${DATA_PATH} \
                            --seed 1 \
                            --momentum 0.9 \
                            --arch ${ARCH} --share-all-embeddings \
                            --optimizer sgd  --clip-norm 0.0 \
                            --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                            --criterion label_smoothed_cross_entropy \
                            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
                            --lr 0.1 --min-lr 1e-9 \
                            --label-smoothing 0.1 --weight-decay 0.0001 \
                            --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                            --update-freq 1 --no-progress-bar --log-interval 50 \
                            --ddp-backend no_c10d \
                            --keep-last-epochs ${NUM} --max-epoch 55 \
                            --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                            | tee -a ${OUTPUT_PATH}/train_log.txt

# --early-stop ${NUM} \

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

PYTHONIOENCODING=utf-8 python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
--target-lang en \
> ${RESULT_PATH}/res.txt


# seed 2

OUTPUT_PATH=log/sgd_2
mkdir -p $OUTPUT_PATH
export CUDA_VISIBLE_DEVICES=0; python main.py ${DATA_PATH} \
                            --seed 2 \
                            --momentum 0.9 \
                            --arch ${ARCH} --share-all-embeddings \
                            --optimizer sgd  --clip-norm 0.0 \
                            --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                            --criterion label_smoothed_cross_entropy \
                            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
                            --lr 0.1 --min-lr 1e-9 \
                            --label-smoothing 0.1 --weight-decay 0.0001 \
                            --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                            --update-freq 1 --no-progress-bar --log-interval 50 \
                            --ddp-backend no_c10d \
                            --keep-last-epochs ${NUM} --max-epoch 55 \
                            --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                            | tee -a ${OUTPUT_PATH}/train_log.txt

# --early-stop ${NUM} \

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

PYTHONIOENCODING=utf-8 python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
--target-lang en \
> ${RESULT_PATH}/res.txt

# seed 3

OUTPUT_PATH=log/sgd_3
mkdir -p $OUTPUT_PATH
export CUDA_VISIBLE_DEVICES=0; python main.py ${DATA_PATH} \
                            --seed 3 \
                            --momentum 0.9 \
                            --arch ${ARCH} --share-all-embeddings \
                            --optimizer sgd  --clip-norm 0.0 \
                            --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                            --criterion label_smoothed_cross_entropy \
                            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
                            --lr 0.1 --min-lr 1e-9 \
                            --label-smoothing 0.1 --weight-decay 0.0001 \
                            --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                            --update-freq 1 --no-progress-bar --log-interval 50 \
                            --ddp-backend no_c10d \
                            --keep-last-epochs ${NUM} --max-epoch 55 \
                            --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                            | tee -a ${OUTPUT_PATH}/train_log.txt

# --early-stop ${NUM} \

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

PYTHONIOENCODING=utf-8 python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
--target-lang en \
> ${RESULT_PATH}/res.txt

# seed 4 

OUTPUT_PATH=log/sgd_4
mkdir -p $OUTPUT_PATH
export CUDA_VISIBLE_DEVICES=0; python main.py ${DATA_PATH} \
                            --seed 4 \
                            --momentum 0.9 \
                            --arch ${ARCH} --share-all-embeddings \
                            --optimizer sgd  --clip-norm 0.0 \
                            --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                            --criterion label_smoothed_cross_entropy \
                            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
                            --lr 0.1 --min-lr 1e-9 \
                            --label-smoothing 0.1 --weight-decay 0.0001 \
                            --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                            --update-freq 1 --no-progress-bar --log-interval 50 \
                            --ddp-backend no_c10d \
                            --keep-last-epochs ${NUM} --max-epoch 55 \
                            --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                            | tee -a ${OUTPUT_PATH}/train_log.txt

# --early-stop ${NUM} \

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

PYTHONIOENCODING=utf-8 python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
--target-lang en \
> ${RESULT_PATH}/res.txt