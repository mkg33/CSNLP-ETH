#!/usr/bin/env bash
# sample script for running predictions in a sequence
set -e


python3 eval_multi.py \
--model-path /Users/piki/Documents/CSNLP/models/multi_style512.pt \
--data-root pan25_data \
--output-dir pred_multi_style512 \

python3 eval_sliced_ot.py \
--model-path /Users/piki/Documents/CSNLP/models/sliced_style512.pt \
--data-root pan25_data \
--output-dir pred_sliced_style512 \

python3 eval_max_slice.py \
--model-path /Users/piki/Documents/CSNLP/models/max_proj_style512.pt \
--data-root pan25_data \
--output-dir pred_max_proj_style512 \
