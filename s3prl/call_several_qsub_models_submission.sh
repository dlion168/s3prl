#!/bin/bash

# Models to test
models=(
    "task_vector_dhubert_ls_960_weight_0.7_and_mert_ls_960_weight_0.3_both_init_hubert_both_same_seed_True_ties_weight_0.4_enable_sign_interference_True_as_ties_True"
    "task_vector_dhubert_ls_960_weight_0.8_and_mert_ls_960_weight_0.3_both_init_hubert_both_same_seed_True_ties_weight_0.4_enable_sign_interference_True_as_ties_True" 
)


    

# Tasks to test
tasks=("asr" "sid" "vocalset_singer_id" "instrument_nsynth" "vocalset_technique_id" "aec_esc50" "pitch_nsynth")


# PBS script to call
pbs_script="paralell-tasks-many-gpus-downstram-qsub-containers.sh"

# GPU allocation per job
gpu_config=(4 2 1)  # Number of GPUs per job

# Initial row values
current_row=153
logfile_row=153

# Submission loop
for model in "${models[@]}"; do
    task_idx=0
    while [ $task_idx -lt ${#tasks[@]} ]; do
        for gpu_count in "${gpu_config[@]}"; do
            if [ $gpu_count -eq 4 ]; then
                select_value=1  # Use 2 nodes
            elif [ $gpu_count -eq 2 ]; then
                select_value=3  # Use 2 node
            else
                select_value=1  # Use 1 node
            fi
            # Batch tasks based on GPU count
            tasks_batch=("${tasks[@]:$task_idx:$gpu_count}")

            echo "Submitting PBS job for model: $model with tasks: ${tasks_batch[*]} and GPUs: $gpu_count"

            # Dynamically set resource requirements and submit PBS job
            qsub \
                -l select=$select_value:ncpus=64:ngpus=$gpu_count \
                -v distilled_model_checkpoint="$model",\
tasks="${tasks_batch[*]}",\
num_gpus=$gpu_count,\
stage="train",\
current_row=$current_row,\
logfile_row=$logfile_row,\
checkpoint_method="hardcoded" \
                $pbs_script

            # Increment task index for the next batch
            task_idx=$((task_idx + gpu_count))
            
            # Break if no more tasks
            if [ $task_idx -ge ${#tasks[@]} ]; then
                break
            fi
        done
    done

    # Increment row values for the next model
    current_row=$((current_row + 1))
    logfile_row=$((logfile_row + 1))
done

echo "All jobs submitted."
