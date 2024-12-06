#!/bin/bash

# Usage example:
# sbatch run_downstream_sbatch.sh <distilled_model_checkpoint> <task> <stage> <current_row> <logfile_row>
# sbatch run_downstream_twcc.sh task_vector_dhubert_ls_100_and_mert_ls_960_both_init_hubert_both_same_seed_using_speech_tsv_weight_0.1 ic train 48 48 hardcoded multi_distiller_local
# sbatch run_downstream_sbatch.sh DistilHuBERT_100hrs_libri_l1_cos pitch_nsynth train 28 26

# sbatch run_downstream_sbatch.sh $model asr
#SBATCH --job-name=evaluate --account=MST113234
#SBATCH -p gp1d
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=0-18:44:00

# Set variables from input arguments or use default values
distilled_model_checkpoint=${1:-"default_checkpoint"}
task=${2:-"ic"}                # Default to "ic" if no task is provided
stage=${3:-"train"}       # Default to "evaluating" if no stage is provided
current_row=${4:-90}           # Default to 90 if no current_row is provided
logfile_row=${5:-90}
checkpoint_method=${6:-"hardcoded"} # or list_based
upstream=${7:-"multi_distiller_local"} # multi_distiller_local distiller_local

log_file="logfiles/downstream/${distilled_model_checkpoint}/${task}/paper_method.log"

#tasks :   instrument_nsynth    pitch_nsynth      aec_esc50 vocalset_singer_id "vocalset_technique_id"   "ks" "ic" "emotion" "sid" "asr"
# Create the log directory if it doesn't exist
mkdir -p "$(dirname "$log_file")"
echo "log at: $log_file"
# Set the output log file for SLURM
#SBATCH --output="$log_file"
export CUDA_VISIBLE_DEVICES=0
# Run the Singularity container and execute the commands
#srun -k --output="$log_file" 
singularity exec --nv --bind  /home/fabian2024/multi_distiller/s3prl/s3prl/:/workspace/s3prl/s3prl,/work/twsgxyc199/sslm-data/:/livingrooms/public/superb2/,\
/work/u8786328/mdd_data/dataset/:/livingrooms/public/superb/,/work/u8786328/dataset/:/livingrooms/public/superb3/,/work/twsgxyc199/:/work/twsgxyc199/ ../../s3prl_for_sslm_v2.sif \
/workspace/s3prl/s3prl/run_inside_container_downstream_twcc.sh "$distilled_model_checkpoint" "$task" "$stage" "$current_row" "$upstream" "$log_file" "$logfile_row" "$checkpoint_method" > "$log_file" 2>&1

echo "task completed"