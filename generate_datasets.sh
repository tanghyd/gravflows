# #!/bin/bash

# # if environment variable DATASET_DIR is not specified
# # export DATASET_DIR=/fred/oz016/datasets/
# # env DATASET_DIR=/fred/oz016/datasets/ bash generate_datasets.sh
# # then we set it to be a default path

dataset_dir=${DATASET_DIR:-"/mnt/datahole/daniel/gravflows/lfigw"}
# dataset_dir=${DATASET_DIR:-"gwpe/datasets"}
echo "Saving datasets to ${dataset_dir}/"

# timer
SECONDS=0

# Sample Parameters from Priors

# reduced basis data set
python -m gwpe.parameters \
    -n 50000 \
    -d "${dataset_dir}/basis/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/fiducial_extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

# training data set
python -m gwpe.parameters \
    -n 1000000 \
    -d "${dataset_dir}/train/" \
    -c gwpe/config_files/intrinsics.ini \
    --overwrite \
    --metadata \
    --verbose
    # -c gwpe/config_files/extrinsics.ini \

# validation data set
python -m gwpe.parameters \
    -n 10000 \
    -d "${dataset_dir}/validation/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

# test data set
python -m gwpe.parameters \
    -n 10000 \
    -d "${dataset_dir}/test/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

## Noise (Power Spectral Density)

# generate PSD files once
psd_out_dir="${dataset_dir}/train/PSD/" 
python -m gwpe.noise \
    -d /mnt/datahole/daniel/gwosc/O1 \
    -s gwpe/config_files/static_args.ini \
    -o "${psd_out_dir}" \
    --verbose \
    --validate
    
# copy PSD files to other dataset partitions for completeness
for partition in "basis" "validation" "test"
do
    for ifo in "H1" "L1"
    do
        cp -r "${dataset_dir}/train/PSD" "${dataset_dir}/${partition}"
    done

done

## Waveforms

# Generate an independent dataset to fit SVD
# Clean waveforms for reduced basis
python -m gwpe.waveforms \
    -d "${dataset_dir}/basis/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/basis/PSD/" \
    --ref_ifo "H1" \
    --lowpass \
    --whiten \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5000 \
    --workers 12 \
    # --ifos "H1" "L1" \
    # --projections_only \
    

# Fit reduced basis with randomized SVD
python -m gwpe.basis \
    -n 600 \
    -d "${dataset_dir}/basis/" \
    -s gwpe/config_files/static_args.ini \
    --file 'projections.npy' \
    --overwrite \
    --verbose \
    --pytorch \
    --concatenate
    # --validate \

# # run waveform generation script for training data
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
    --workers 12
    # --add_noise \
    # --gaussian \
    # --projections_only \
    # --ifos "H1" "L1" \

# pre-generate coeffficients for training
python -m gwpe.basis \
    -n 100 \
    -s gwpe/config_files/static_args.ini \
    --basis_dir "${dataset_dir}/basis/" \
    --data_dir "${dataset_dir}/train/" \
    --file 'waveforms.npy' \
    --concatenate \
    --coefficients \
    --overwrite \
    --verbose \
    --chunk_size 5000 \
    --pytorch \
    --concatenate
    # --validate \

# Evaluation datasets

# validation and test waveforms are noisy
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
    --workers 12
    # --add_noise \
    # --gaussian \
    # --projections_only \
    
python -m gwpe.basis \
    -n 100 \
    -s gwpe/config_files/static_args.ini \
    --basis_dir "${dataset_dir}/basis/" \
    --data_dir "${dataset_dir}/validation/" \
    --file 'waveforms.npy' \
    --concatenate \
    --coefficients \
    --overwrite \
    --verbose \
    --chunk_size 5000
    # --validate \

# validation and test waveforms are noisy
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
    --workers 12
    # --projections_only \

python -m gwpe.basis \
    -n 100 \
    -s gwpe/config_files/static_args.ini \
    --basis_dir "${dataset_dir}/basis/" \
    --data_dir "${dataset_dir}/test/" \
    --file 'waveforms.npy' \
    --coefficients \
    --overwrite \
    --verbose \
    --chunk_size 5000
    # --validate \

# print out size of datasets
for partition in "train" "basis" "validation" "test"
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