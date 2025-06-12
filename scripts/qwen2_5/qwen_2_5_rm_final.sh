#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 这个脚本用于训练奖励模型，同时要能够评测模型的效果
MODEL_NAME_OR_PATH="./Qwen2.5-0.5B-Instruct" # model path

TRAIN_DATASETS="../../assets/text_to_text/hw" # rm dataset path
TRAIN_TEMPLATE="HOMEWORK" # dataset template
TRAIN_SPLIT="train" # split the dataset

EVAL_DATASETS="../../assets/text_to_text/hw"
EVAL_TEMPLATE="HOMEWORK"
EVAL_NAME="analysis"
EVAL_SPLIT="validation"

OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR

if [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "OUTPUT_ROOT_DIR is not set"
    OUTPUT_ROOT_DIR="../outputs"
fi

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5_rm_final_3epoch" # output dir

# For wandb online logging
# export WANDB_API_KEY="6bfb185b56a3b9fa23a5fd706239d8c25ac9f957"
export WANDB_API_KEY=""

# Source the setup script
source ../setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_split ${TRAIN_SPLIT} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_name ${EVAL_NAME} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --epochs 3