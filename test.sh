#!/bin/bash

dataset_dir=${DATASET_DIR:-"datasets"}
partition="train"
psd_out_dir="${dataset_dir}/${partition}/PSD/"
echo "$psd_out_dir"