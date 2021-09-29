# #!/bin/bash

# # if environment variable DATASET_DIR is not specified
# # export DATASET_DIR=/fred/oz016/datasets/
# # env DATASET_DIR=/fred/oz016/datasets/ bash generate_datasets.sh
# # then we set it to be a default path

# dataset_dir=${DATASET_DIR:-"/mnt/datahole/daniel/gravflows/timing"}
dataset_dir=${DATASET_DIR:-"gwpe/datasets"}
echo "Saving datasets to ${dataset_dir}/"

# timer
SECONDS=0

# training data set
python -m gwpe.parameters \
    -n 10000 \
    -d "${dataset_dir}/train/" \
    -c gwpe/config_files/intrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

# reduced basis data set
python -m gwpe.parameters \
    -n 5000 \
    -d "${dataset_dir}/basis/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

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

# run (intrinsic) waveform generation script for training data
python -m gwpe.waveforms \
    -d "${dataset_dir}/train/" \
    -s gwpe/config_files/static_args.ini \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5000 \
    --workers 12

# generate PSD files once
psd_out_dir="${dataset_dir}/train/PSD/" 
python -m gwpe.noise \
    -d /mnt/datahole/daniel/gwosc/O1 \
    -s gwpe/config_files/static_args.ini \
    -o "${psd_out_dir}" \
    --verbose \
    --validate
    
# project waveforms for validation and test
for partition in "basis" "validation" "test"
do

    # copy PSD files to other dataset partitions for completeness
    for ifo in "H1" "L1"
    do
        # psd_out_dir="${dataset_dir}/${partition}"
        # mkdir -p "${psd_out_dir}"
        cp -r "${dataset_dir}/train/PSD" "${dataset_dir}/${partition}"
    done

done

# generate clean waveforms for reduced basis
python -m gwpe.waveforms \
    -d "${dataset_dir}/basis/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/basis/PSD/" \
    --ifos "H1" "L1" \
    --bandpass \
    --whiten \
    --projections_only \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --downcast \
    --chunk_size 5000 \
    --workers 12
    
# fit reduced basis with randomized SVD
python -m gwpe.basis \
    -n 1000 \
    -d "${dataset_dir}/basis/" \
    -s gwpe/config_files/static_args.ini \
    -f reduced_basis.npy \
    --overwrite \
    --verbose \
    --validate \
    --cuda 

# validation and test waveforms are noisy
python -m gwpe.waveforms \
    -d "${dataset_dir}/validation/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/validation/PSD/" \
    --ifos "H1" "L1" \
    --bandpass \
    --whiten \
    --projections_only \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --downcast \
    --chunk_size 5000 \
    --workers 12
    
# validation and test waveforms are noisy
python -m gwpe.waveforms \
    -d "${dataset_dir}/test/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/test/PSD/" \
    --ifos "H1" "L1" \
    --add_noise \
    --bandpass \
    --whiten \
    --projections_only \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --downcast \
    --chunk_size 5000 \
    --workers 12

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