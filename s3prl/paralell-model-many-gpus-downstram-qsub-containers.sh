#!/bin/bash
#PBS -P 13003882
#PBS -j oe
#PBS -N TASK_VECTOR
#PBS -q normal
#PBS -l select=2:ncpus=64:ngpus=2
#PBS -l walltime=18:43:58
#PBS -m ae
#PBS -M fabian.acustica@gmail.com
#PBS -o /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl/logfiles/downstream/testing-all.out

# Set variables from input arguments or use default values
# Set variables from environment variables (passed via `-v` in `qsub`)
# qsub -v stage="train",checkpoint_method="list_based" paralell-model-many-gpus-downstram-qsub-containers.sh

stage=${stage:-"train"}
checkpoint_method=${checkpoint_method:-"hardcoded"}

upstream="distiller_local"  # Use this project ID: 13003882

# Define the task and model checkpoints
task="aec_esc50"  # "vocalset_singer_id" "vocalset_technique_id"   "ks" "ic" "instrument_nsynth"  "pitch_nsynth"  "aec_esc50"
declare -A model_info

# Add your model checkpoints and their corresponding current_row and logfile_row
model_info["distilhubert-ls960-own"]="current_row=27 logfile_row=25"
model_info["DistilHuBERT_100hrs_libri_l1_cos"]="current_row=28 logfile_row=26"

# Extract the list of model checkpoints
distilled_model_checkpoints=("${!model_info[@]}")


# Define log directory
log_dir="/home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl/logfiles/downstream/${task}"

# Create log directory if it doesn't exist
mkdir -p "$log_dir"
echo "Logs will be stored in: $log_dir"

# Load Singularity module
module load singularity

# Change to the working directory
cd $PBS_O_WORKDIR
echo "Current directory is $PBS_O_WORKDIR"

# Loop over model checkpoints and launch each on a separate GPU
for i in "${!distilled_model_checkpoints[@]}"; do
    distilled_model_checkpoint="${distilled_model_checkpoints[$i]}"

    # Extract current_row and logfile_row for this model checkpoint
    eval "${model_info[$distilled_model_checkpoint]}"

    log_file="$log_dir/${distilled_model_checkpoint}_paper_method.log"

    echo "Starting task: $task with model checkpoint: $distilled_model_checkpoint on GPU $i"
    echo "Log file: $log_file"
    echo "Using current_row: $current_row and logfile_row: $logfile_row"

    # Run the task in the background, assigning a different GPU to each job
    export CUDA_VISIBLE_DEVICES=$i
    singularity exec --nv \
        --bind /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl:/workspace/s3prl/s3prl,\
/home/project/13003821/fabian/corpus/superb:/livingrooms/public/superb,\
/home/project/13003821/fabian/projects:/home/project/13003821/fabian/projects \
        ../../s3prl_for_sslm_v2.sif \
        /workspace/s3prl/s3prl/run_inside_container_downstream.sh \
        "$distilled_model_checkpoint" "$task" "$stage" "$current_row" "$upstream" "$log_file" "$logfile_row" "$checkpoint_method" \
        > "$log_file" 2>&1 &
done

# Wait for all background tasks to complete
wait
echo "All tasks completed."
exit 0
