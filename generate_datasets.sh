# #!/bin/bash

# # if environment variable DATASET_DIR is not specified
# # export DATASET_DIR=/fred/oz016/datasets/
# # env DATASET_DIR=/fred/oz016/datasets/ bash generate_datasets.sh
# # then we set it to be a default path

# dataset_dir=${DATASET_DIR:-"<PATH TO OUTPUT DATASET DIR>"}

dataset_dir=${DATASET_DIR:-"data"}
echo "Saving datasets to ${dataset_dir}/"

# timer
SECONDS=0

# Sample Parameters from Priors

# training data set
python -m gwpe.parameters \
    -n 10000 \
    -d "${dataset_dir}/train/" \
    -c gwpe/config_files/intrinsics.ini \
    --overwrite \
    --metadata \
    --verbose
    # -c gwpe/config_files/extrinsics.ini \

# validation data set
python -m gwpe.parameters \
    -n 1000 \
    -d "${dataset_dir}/validation/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

# test data set
python -m gwpe.parameters \
    -n 1000 \
    -d "${dataset_dir}/test/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

## Noise (Power Spectral Density)

# # generate PSD files once
# psd_out_dir="${dataset_dir}/train/PSD/" 
# python -m gwpe.noise \
#     -d /mnt/datahole/daniel/gwosc/O1 \
#     -s gwpe/config_files/static_args.ini \
#     -o "${psd_out_dir}" \
#     --verbose \
#     --validate
    
# # copy PSD files to other dataset partitions for completeness
# for partition in "validation" "test"
# do
#     for ifo in "H1" "L1"
#     do
#         cp -r "${dataset_dir}/train/PSD" "${dataset_dir}/${partition}"
#     done

# done

# typically we would estimate a PSD via welch estimate on real strain
# due to data location issues we include a sample PSD for testing
# copy sample PSD to dataset partition locations
for partition in "train" "validation" "test"
do
    for ifo in "H1" "L1"
    do
        cp -r "PSD" "${dataset_dir}/${partition}"
    done

done

## Waveforms
# run waveform generation script for training data
python -m gwpe.waveforms \
    -d "${dataset_dir}/train/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/train/PSD/" \
    --ref_ifo "H1" \
    --lowpass \
    --whiten \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 10000 \
    --workers 4

# Evaluation datasets

# validation set
python -m gwpe.waveforms \
    -d "${dataset_dir}/validation/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/validation/PSD/" \
    --ifos "H1" "L1" \
    --ref_ifo "H1" \
    --lowpass \
    --whiten \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5000 \
    --workers 4
    # --add_noise \
    # --gaussian \
    # --projections_only \
    
# test waveforms are noisy
python -m gwpe.waveforms \
    -d "${dataset_dir}/test/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/test/PSD/" \
    --ifos "H1" "L1" \
    --ref_ifo "H1" \
    --add_noise \
    --gaussian \
    --lowpass \
    --whiten \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5000 \
    --workers 4
    # --projections_only \

# print out size of datasets
for partition in "train" "validation" "test"
do
    echo "$(du -sh ${dataset_dir}/${partition}/)"
done

# print out runtime
if (( $SECONDS > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)."
elif (( $SECONDS > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)."
else
    echo "Completed in $SECONDS seconds."
fi