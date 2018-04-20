#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/elguinds/cuda/lib64

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Training iterations.
NUM_ITERATIONS=10000
DATE=""
CROP_SIZE=256
TRAIN_SIZE=16
ATROUS_1=6
ATROUS_2=12
ATROUS_3=18
OUT_STRIDE=16
BATCH_NORM="True"

# Set up the working directories.

while [  $NUM_ITERATIONS -lt 20001 ]; do

    PASCAL_FOLDER="bladder"
    EXP_FOLDER="exp/axial${DATE}"
    TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
    EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
    VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
    EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
    mkdir -p "${TRAIN_LOGDIR}"
    mkdir -p "${EVAL_LOGDIR}"
    mkdir -p "${VIS_LOGDIR}"
    mkdir -p "${EXPORT_DIR}"


    PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord_ax"

    python "${WORK_DIR}"/train.py \
      --logtostderr \
      --train_split="train_ax" \
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1}" \
      --atrous_rates="${ATROUS_2}" \
      --atrous_rates="${ATROUS_3}" \
      --output_stride="${OUT_STRIDE}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_batch_size="${TRAIN_SIZE}" \
      --training_number_of_steps="${NUM_ITERATIONS}" \
      --tf_initial_checkpoint="${WORK_DIR}/${DATASET_DIR}/start_weights/model.ckpt" \
      --train_logdir="${TRAIN_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --dataset="${PASCAL_FOLDER}"

    # Visualize the results.
    python "${WORK_DIR}"/vis.py \
      --logtostderr \
      --vis_split="val_ax" \
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
      --eval_split="val_ax" \
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
      --crop_size="${CROP_SIZE}" \
      --inference_scales=1.0

    ########################################################
    ########################################################

    # Set up the working directories.
    EXP_FOLDER="exp/saggital${DATE}"
    TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
    EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
    VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
    EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
    mkdir -p "${TRAIN_LOGDIR}"
    mkdir -p "${EVAL_LOGDIR}"
    mkdir -p "${VIS_LOGDIR}"
    mkdir -p "${EXPORT_DIR}"


    PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord_sag"

    # Train 10 iterations.
    python "${WORK_DIR}"/train.py \
      --logtostderr \
      --train_split="train_sag" \
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1}" \
      --atrous_rates="${ATROUS_2}" \
      --atrous_rates="${ATROUS_3}" \
      --output_stride="${OUT_STRIDE}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_batch_size="${TRAIN_SIZE}" \
      --training_number_of_steps="${NUM_ITERATIONS}" \
      --tf_initial_checkpoint="${WORK_DIR}/${DATASET_DIR}/start_weights/model.ckpt" \
      --train_logdir="${TRAIN_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --dataset="${PASCAL_FOLDER}"

    # Visualize the results.
    python "${WORK_DIR}"/vis.py \
      --logtostderr \
      --vis_split="val_sag" \
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
      --eval_split="val_sag" \
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
      --crop_size="${CROP_SIZE}" \
      --inference_scales=1.0

    ########################################################
    ########################################################

    # Set up the working directories.
    EXP_FOLDER="exp/coronal${DATE}"
    TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
    EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
    VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
    EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
    mkdir -p "${TRAIN_LOGDIR}"
    mkdir -p "${EVAL_LOGDIR}"
    mkdir -p "${VIS_LOGDIR}"
    mkdir -p "${EXPORT_DIR}"


    PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord_cor"

    # Train 10 iterations.
    python "${WORK_DIR}"/train.py \
      --logtostderr \
      --train_split="train_cor" \
      --model_variant="xception_65" \
      --atrous_rates="${ATROUS_1}" \
      --atrous_rates="${ATROUS_2}" \
      --atrous_rates="${ATROUS_3}" \
      --output_stride="${OUT_STRIDE}" \
      --decoder_output_stride=4 \
      --fine_tune_batch_norm="${BATCH_NORM}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_crop_size="${CROP_SIZE}" \
      --train_batch_size="${TRAIN_SIZE}"  \
      --training_number_of_steps="${NUM_ITERATIONS}" \
      --tf_initial_checkpoint="${WORK_DIR}/${DATASET_DIR}/start_weights/model.ckpt" \
      --train_logdir="${TRAIN_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --dataset="${PASCAL_FOLDER}"

    # Visualize the results.
    python "${WORK_DIR}"/vis.py \
      --logtostderr \
      --vis_split="val_cor" \
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
      --eval_split="val_cor" \
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
      --crop_size="${CROP_SIZE}" \
      --inference_scales=1.0

    NUM_ITERATIONS=$((NUM_ITERATIONS + 10000))

done