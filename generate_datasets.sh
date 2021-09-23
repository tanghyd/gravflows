# #!/bin/bash

# # if environment variable DATASET_DIR is not specified
# # export DATASET_DIR=/fred/oz016/datasets/
# # env DATASET_DIR=/fred/oz016/datasets/ bash generate_datasets.sh
# # then we set it to be a default path

# dataset_dir=${DATASET_DIR:-"/mnt/datahole/daniel/gravflows/datasets"}
dataset_dir=${DATASET_DIR:-"datasets"}
echo "Saving datasets to ${dataset_dir}/" 

# training data set
python generate_parameters.py \
    -n 10000 \
    -d "${dataset_dir}/train/" \
    -c config_files/intrinsics.ini \
    --overwrite \
    --metadata

# reduced basis data set
python generate_parameters.py \
    -n 5000 \
    -d "${dataset_dir}/basis/" \
    -c config_files/intrinsics.ini \
    -c config_files/extrinsics.ini \
    --overwrite \
    --metadata

# validation data set
python generate_parameters.py \
    -n 1000 \
    -d "${dataset_dir}/validation/" \
    -c config_files/intrinsics.ini \
    -c config_files/extrinsics.ini \
    --overwrite \
    --metadata

# test data set
python generate_parameters.py \
    -n 1000 \
    -d "${dataset_dir}/test/" \
    -c config_files/intrinsics.ini \
    -c config_files/extrinsics.ini \
    --overwrite \
    --metadata

# run (intrinsic) waveform generation script for training data
python generate_waveforms.py \
    -d "${dataset_dir}/train/" \
    -s config_files/static_args.ini \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5000 \
    --workers 12

# generate PSD files once
psd_out_dir="${dataset_dir}/train/PSD/" 
python generate_psd.py \
    -d /mnt/datahole/daniel/gwosc/O1 \
    -s config_files/static_args.ini \
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

    if [[ "${partition}" == "basis" ]]; then
        # generate clean waveforms for reduced basis
        python generate_waveforms.py \
            -d "${dataset_dir}/${partition}/" \
            -s config_files/static_args.ini \
            --psd_dir "${dataset_dir}/${partition}/PSD/" \
            --projections_only \
            --ifos "H1" "L1" \
            --overwrite \
            --verbose \
            --validate \
            --metadata \
            --chunk_size 5000 \
            --workers 12
            
        # fit reduced basis with randomized SVD
        python generate_reduced_basis.py \
            -d "${dataset_dir}/${partition}/" \
            -p "${dataset_dir}/${partition}/PSD" \
            -s config_files/static_args.ini \
            -f reduced_basis.npy \
            --ifos "H1" "L1" \
            --bandpass \
            --whiten \
            --overwrite \
            --verbose \

        
    else
        # validation and test waveforms are noisy
        python generate_waveforms.py \
            -d "${dataset_dir}/${partition}/" \
            -s config_files/static_args.ini \
            --psd_dir "${dataset_dir}/${partition}/PSD/" \
            --add_noise \
            --projections_only \
            --overwrite \
            --verbose \
            --validate \
            --metadata \
            --ifos "H1" "L1" \
            --chunk_size 5000 \
            --workers 12
    fi

done


# # to do:
# # print out size of saved datasets in MB?