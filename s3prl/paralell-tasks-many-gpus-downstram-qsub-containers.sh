#!/bin/bash
#PBS -P 13003821
#PBS -j oe
#PBS -N TASK_VECTOR
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=19:43:58
#PBS -m ae
#PBS -M fabian.acustica@gmail.com
#PBS -o /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl/logfiles/downstream/testing-all.out

# Usage example: 
# qsub run_downstream_pbs.sh <distilled_model_checkpoint> <task> <stage> <current_row>
# qsub -v distilled_model_checkpoint="mert-base-hubert-base-equal-weight-merging-correlation",stage="train",current_row=130,logfile_row=130,checkpoint_method="custom",custom_checkpoint="result/merged_pretrain_upstream/permutation-covariance/baseline/test_match_tensors_permute_ff+attn_hubert_base_mert_base_model_average_both_models_equal_weight.ckpt" paralell-tasks-many-gpus-downstram-qsub-containers.sh

# Set variables from input arguments or use default values
# Set variables from environment variables (passed via `-v` in `qsub`)
distilled_model_checkpoint=${distilled_model_checkpoint:-"default_checkpoint"} #### tasks to run are: 
stage=${stage:-"train"}
current_row=${current_row:-90}
logfile_row=${logfile_row:-90}
checkpoint_method=${checkpoint_method:-"hardcoded"}
custom_checkpoint=${custom_checkpoint:-""}

#upstream="multi_distiller_local"  ### USE THIS OTHER PROJECT ID ALSO : 13003882
upstream="hubert_local"

# Define tasks and log file paths
tasks=("aec_esc50" "vocalset_singer_id" "vocalset_technique_id" "ks")  # "vocalset_singer_id" "vocalset_technique_id"   "ks" "ic" "instrument_nsynth"  "pitch_nsynth"  "aec_esc50"
log_dir="/home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl/logfiles/downstream/${distilled_model_checkpoint}"

# Create log directories if they don't exist
mkdir -p "$log_dir"
echo "Logs will be stored in: $log_dir"


# Set the output log file for SLURM
module load singularity

cd $PBS_O_WORKDIR
echo "current dir is $PBS_O_WORKDIR"
# Loop over tasks and launch each task on a separate GPU
for i in {0..3}; do
#i=0
    task="${tasks[$i]}"
    log_file="$log_dir/${task}_paper_method.log"

    echo "Starting task: $task on GPU $i with log file: $log_file"

    # Run the task in the background, assigning a different GPU to each job
    export CUDA_VISIBLE_DEVICES=$i
    CUDA_VISIBLE_DEVICES=$i singularity exec --nv --bind /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl:/workspace/s3prl/s3prl,\
/home/project/13003821/fabian/corpus/superb:/livingrooms/public/superb,/home/project/13003821/fabian/projects:/home/project/13003821/fabian/projects ../../s3prl_for_sslm_v2.sif \
    /workspace/s3prl/s3prl/run_inside_container_downstream.sh "$distilled_model_checkpoint" "$task" "$stage" "$current_row" "$upstream" "$log_file" "$logfile_row" "$checkpoint_method" "$custom_checkpoint" > "$log_file" 2>&1 &
done

# Wait for all background tasks to complete
wait
echo "All tasks completed."
exit 0