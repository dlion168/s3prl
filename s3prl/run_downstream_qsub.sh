#!/bin/bash
#PBS -P 13003821
#PBS -j oe
#PBS -N TASK_VECTOR
#PBS -q normal
#PBS -l select=1:ncpus=48:ngpus=1
#PBS -l walltime=09:23:58
#PBS -m ae
#PBS -M fabian.acustica@gmail.com
#PBS -o /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl/logfiles/downstream/testing.out

# Usage example: 
# qsub run_downstream_pbs.sh <distilled_model_checkpoint> <task> <stage> <current_row>
# qsub -v distilled_model_checkpoint="task_vector_dhubert_960_and_mert_only_music4all_data_and_mert_init_music_tsv_weight_0.2",task="vocalset_singer_id",stage="train",current_row=78,logfile_row=78,cont="" run_downstream_qsub.sh
# Set variables from input arguments or use default values
# Set variables from environment variables (passed via `-v` in `qsub`)
distilled_model_checkpoint=${distilled_model_checkpoint:-"default_checkpoint"}
task=${task:-"ic"}
stage=${stage:-"train"}
current_row=${current_row:-90}
cont=${cont:-""}


cd $PBS_O_WORKDIR

log_file="logfiles/downstream/${distilled_model_checkpoint}/${task}_paper_method${cont}.log"
upstream="multi_distiller_local"
#upstream="distiller_local"
export CUDA_VISIBLE_DEVICES=0

# Create the log directory if it doesn't exist
mkdir -p "$(dirname "$log_file")"
echo "log at: $log_file"
# Set the output log file for SLURM
module load singularity
# Run the Singularity container and execute the commands
CUDA_VISIBLE_DEVICES=0 singularity exec --nv --bind /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl:/workspace/s3prl/s3prl,\
/home/project/13003821/fabian/corpus/superb:/livingrooms/public/superb ../../s3prl_for_sslm_v2.sif \
/workspace/s3prl/s3prl/run_inside_container_downstream.sh "$distilled_model_checkpoint" "$task" "$stage" "$current_row" "$upstream" "$log_file" "$logfile_row" > "$log_file" 2>&1

echo "task completed"