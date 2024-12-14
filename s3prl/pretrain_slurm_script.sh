#!/bin/bash

#SBATCH -p 10d                                  # Partition (queue) to submit to
#SBATCH --nodelist=s04                          # Node to submit to
#SBATCH --job-name=mert_pretrain
#SBATCH -n 1                                    # Number of tasks
#SBATCH --gres=gpu:RTX6000AdaGeneration:1       # Request 2 GPUs
#SBATCH --cpus-per-task=24                      # Number of CPU cores per task
#SBATCH --time=1-23:00:00                          # Time limit (4 hours)
#SBATCH --output logfiles/test_job_%j.out
#SBATCH --error logfiles/test_job_%j.err
# Load the Singularity environment and execute the job
logfile="logfiles/pretrain/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

singularity exec --nv \
    --bind /livingrooms/fabian/projects/distillation-sfm/s3prl/s3prl:/workspace/s3prl/s3prl,\
/groups/ycevan/datasets:/workspace/audio_data,\
/groups/public/benchmark:/workspace/superb_data1,\
/livingrooms/public/superb:/livingrooms/public/superb,\
/livingrooms/fabian/music4all:/workspace/music4all \
../../s3prl_for_sslm_v2.sif \
bash -c "
    source /opt/conda/bin/activate s3prl_old_cuda;
    BASE_DIR_S3PRL='/workspace/s3prl/s3prl';
    cd \$BASE_DIR_S3PRL;
    pip install nnAudio;
    gpus=1;
    model=distill_mert_init_mert_music4all_avgpool;
    current_row=28;
    config_file='pretrain/multi_distiller/config_model.yaml';
    upstream='multi_distiller';
    export CUDA_VISIBLE_DEVICES=0;
    python run_pretrain.py -u \$upstream -g \$config_file -n \$model --logfile \"${logfile}\" --current_row \$current_row --json_file ./results-for-dd-research-f18106ee2c51.json
"