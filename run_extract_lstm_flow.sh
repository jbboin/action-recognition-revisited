#!/usr/bin/env bash

DATASET_DIR="$(python -c 'import config;print(config.DATASET_DIR)')"
LRCN_MODELS_DIR="$(python -c 'import config;print(config.LRCN_MODELS_DIR)')"

python run_extract.py \
--model $LRCN_MODELS_DIR/flow_lstm_model_iter_50000.caffemodel \
--save_folder $DATASET_DIR/extracted_features_lstm_flow \
--im_path $DATASET_DIR/flow_images \
--flow
