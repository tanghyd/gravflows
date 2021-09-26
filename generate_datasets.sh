# #!/bin/bash

# # if environment variable DATASET_DIR is not specified
# # export DATASET_DIR=/fred/oz016/datasets/
# # env DATASET_DIR=/fred/oz016/datasets/ bash generate_datasets.sh
# # then we set it to be a default path

dataset_dir=${DATASET_DIR:-"/mnt/datahole/daniel/gravflows/datasets"}
# dataset_dir=${DATASET_DIR:-"datasets"}
echo "Saving datasets to ${dataset_dir}/"

# timer
SECONDS=0

# # training data set
# python generate_parameters.py \
#     -n 1000000 \
#     -d "${dataset_dir}/train/" \
#     -c config_files/intrinsics.ini \
#     --overwrite \
#     --metadata

# # reduced basis data set
# python generate_parameters.py \
#     -n 100000 \
#     -d "${dataset_dir}/basis/" \
#     -c config_files/intrinsics.ini \
#     -c config_files/extrinsics.ini \
#     --overwrite \
#     --metadata

# # validation data set
# python generate_parameters.py \
#     -n 10000 \
#     -d "${dataset_dir}/validation/" \
#     -c config_files/intrinsics.ini \
#     -c config_files/extrinsics.ini \
#     --overwrite \
#     --metadata

# # test data set
# python generate_parameters.py \
#     -n 10000 \
#     -d "${dataset_dir}/test/" \
#     -c config_files/intrinsics.ini \
#     -c config_files/extrinsics.ini \
#     --overwrite \
#     --metadata

# # run (intrinsic) waveform generation script for training data
# python generate_waveforms.py \
#     -d "${dataset_dir}/train/" \
#     -s config_files/static_args.ini \
#     --overwrite \
#     --verbose \
#     --validate \
#     --metadata \
#     --chunk_size 5000 \
#     --workers 12

# # generate PSD files once
# psd_out_dir="${dataset_dir}/train/PSD/" 
# python generate_psd.py \
#     -d /mnt/datahole/daniel/gwosc/O1 \
#     -s config_files/static_args.ini \
#     -o "${psd_out_dir}" \
#     --verbose \
#     --validate
    
# # project waveforms for validation and test
# for partition in "basis" "validation" "test"
# do

#     # copy PSD files to other dataset partitions for completeness
#     for ifo in "H1" "L1"
#     do
#         # psd_out_dir="${dataset_dir}/${partition}"
#         # mkdir -p "${psd_out_dir}"
#         cp -r "${dataset_dir}/train/PSD" "${dataset_dir}/${partition}"
#     done

# done

# # generate clean waveforms for reduced basis
# python generate_waveforms.py \
#     -d "${dataset_dir}/basis/" \
#     -s config_files/static_args.ini \
#     --psd_dir "${dataset_dir}/basis/PSD/" \
#     --ifos "H1" "L1" \
#     --bandpass \
#     --whiten \
#     --projections_only \
#     --overwrite \
#     --verbose \
#     --validate \
#     --metadata \
#     --chunk_size 5000 \
#     --workers 12
    
# # fit reduced basis with randomized SVD
# python generate_reduced_basis.py \
#     -n 1000 \
#     -d "${dataset_dir}/basis/" \
#     -s config_files/static_args.ini \
#     -f reduced_basis.npy \
#     --overwrite \
#     --verbose \
#     --validate \
#     --cuda 

# validation and test waveforms are noisy
python generate_waveforms.py \
    -d "${dataset_dir}/validation/" \
    -s config_files/static_args.ini \
    --psd_dir "${dataset_dir}/validation/PSD/" \
    --ifos "H1" "L1" \
    --add_noise \
    --gaussian \
    --bandpass \
    --whiten \
    --projections_only \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5000 \
    --workers 12

    
# validation and test waveforms are noisy
python generate_waveforms.py \
    -d "${dataset_dir}/test/" \
    -s config_files/static_args.ini \
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