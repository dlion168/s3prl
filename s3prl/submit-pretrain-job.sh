#!/bin/bash
#SBATCH --partition=10d                # Partition name
#SBATCH --nodelist=s06                 # Specify the node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --gres=gpu:RTX6000AdaGeneration:1 # GPU type and count
#SBATCH --cpus-per-task=24             # Number of CPU cores
#SBATCH --mem=28G                      # Memory allocation
#SBATCH --job-name=pretrain            # Job name
#SBATCH --time=4-4:00:00               # Time limit (4 days)
#SBATCH --output=logs/%x_%j.out        # Output log file
#SBATCH --error=logs/%x_%j.err         # Error log file

# Load Singularity and set up bindings
module load singularity
SINGULARITY_IMG=../../s3prl_for_sslm_v2.sif
BINDINGS="
  --bind /livingrooms/fabian/projects/distillation-sfm/s3prl/s3prl:/workspace/s3prl/s3prl,
  /groups/ycevan/datasets:/workspace/audio_data,
  /groups/public/benchmark/LibriSpeech/:/workspace/LibriSpeech,
  /livingrooms/fabian/music4all:/workspace/music4all,
  /livingrooms/fabian/AudioSet:/workspace/AudioSet,
  /groups/public/benchmark:/workspace/superb_data1,
  /livingrooms/public/superb:/livingrooms/public/superb
"

# Run the container and execute the script
singularity exec --nv $BINDINGS $SINGULARITY_IMG bash <<EOF
#!/bin/bash
source /opt/conda/bin/activate s3prl_old_cuda
BASE_DIR_S3PRL="/workspace/s3prl/s3prl"
cd $BASE_DIR_S3PRL
pip install nnAudio

# Uncomment if needed for every run:
 cd /workspace
 git clone https://github.com/huggingface/transformers.git
 cd transformers
 pip install -e .
 cd $BASE_DIR_S3PRL

gpus=1
model=distilhubert_music4all_and_ls960_2layers
logfile=logfiles/pretrain/$model
current_row=31
config_file="pretrain/multi_distiller/config_model.yaml"
upstream="multi_distiller"
export CUDA_VISIBLE_DEVICES=0

nohup python run_pretrain.py -u \$upstream -g \$config_file -n \$model --logfile \$logfile --current_row \$current_row --json_file ./results-for-dd-research-f18106ee2c51.json > \$logfile 2>&1 &
EOF