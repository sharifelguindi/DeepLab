#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/DeepLab directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/elguinds/cuda/lib64

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Standard Input parameters
CROP_SIZE=256
TRAIN_SIZE=12
ATROUS_1=12
ATROUS_2=24
ATROUS_3=36
OUT_STRIDE=8
ATROUS_1_TRAIN=6
ATROUS_2_TRAIN=12
ATROUS_3_TRAIN=18
OUT_STRIDE_TRAIN=16
BASE_LEARNING_RATE=0.007
BATCH_NORM="True"

## Logging changes
DATE="OS_16"
NUM_ITERATIONS=12000
PASCAL_FOLDER="bladder"
EXP_FOLDER="exp/axial${DATE}"
TF_RECORD="tfrecord_ax"
train_set="train_ax"
val_set="val_ax"
max_iterations=40001
print_results=1000

# Loop through total interations

while [  $NUM_ITERATIONS -lt $max_iterations ]; do


    TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
    EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
    VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
    EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
    mkdir -p "${TRAIN_LOGDIR}"
    mkdir -p "${EVAL_LOGDIR}"
    mkdir -p "${VIS_LOGDIR}"
    mkdir -p "${EXPORT_DIR}"


    PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${TF_RECORD}"

    python "${WORK_DIR}"/train.py \
      --logtostderr \
      --train_split="${train_set}"\
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1_TRAIN}" \
      --atrous_rates="${ATROUS_2_TRAIN}" \
      --atrous_rates="${ATROUS_3_TRAIN}" \
      --output_stride="${OUT_STRIDE_TRAIN}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_batch_size="${TRAIN_SIZE}" \
      --training_number_of_steps="${NUM_ITERATIONS}" \
      --base_learning_rate="${BASE_LEARNING_RATE}" \
      --tf_initial_checkpoint="${WORK_DIR}/${DATASET_DIR}/start_weights/model.ckpt" \
      --train_logdir="${TRAIN_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --dataset="${PASCAL_FOLDER}"

    # Visualize the results.
    python "${WORK_DIR}"/vis.py \
      --logtostderr \
      --vis_split="${val_set}" \
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1}" \
      --atrous_rates="${ATROUS_2}" \
      --atrous_rates="${ATROUS_3}" \
      --output_stride="${OUT_STRIDE}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --vis_crop_size="${CROP_SIZE}" \
      --vis_crop_size="${CROP_SIZE}" \
      --checkpoint_dir="${TRAIN_LOGDIR}" \
      --vis_logdir="${VIS_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --max_number_of_iterations=1 \
      --dataset="${PASCAL_FOLDER}"

    python "${WORK_DIR}"/eval.py \
      --logtostderr \
      --eval_split="${val_set}"\
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1}" \
      --atrous_rates="${ATROUS_2}" \
      --atrous_rates="${ATROUS_3}" \
      --output_stride="${OUT_STRIDE}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --eval_crop_size="${CROP_SIZE}" \
      --eval_crop_size="${CROP_SIZE}" \
      --checkpoint_dir="${TRAIN_LOGDIR}" \
      --eval_logdir="${EVAL_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --dataset="${PASCAL_FOLDER}" \
      --max_number_of_evaluations=1 \

    ## Export the trained checkpoint.
    CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
    EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

    python "${WORK_DIR}"/export_model.py \
      --logtostderr \
      --checkpoint_path="${CKPT_PATH}" \
      --export_path="${EXPORT_PATH}" \
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1}" \
      --atrous_rates="${ATROUS_2}" \
      --atrous_rates="${ATROUS_3}" \
      --output_stride="${OUT_STRIDE}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --num_classes=2 \
      --crop_size="${CROP_SIZE}" \
      --crop_size="${CROP_SIZE}"

    NUM_ITERATIONS=$((NUM_ITERATIONS + $print_results))

done