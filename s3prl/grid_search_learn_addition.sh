#!/bin/bash
source /opt/conda/bin/activate s3prl_old_cuda
BASE_DIR_S3PRL="/workspace/s3prl/s3prl"
cd $BASE_DIR_S3PRL
export PYTHONPATH=/workspace/s3prl:$PYTHONPATH

weights=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Install necessary Python packages
pip install scipy==1.5.4 librosa==0.8.0 scikit-learn==0.24.2 matplotlib==3.3.4 modelscope==1.11.0

# Configure Git and pull the latest changes if necessary
cd /workspace/s3prl
git config --global --add safe.directory /workspace/s3prl

cd $BASE_DIR_S3PRL

for weight in "${weights[@]}"; do
    echo "Processing weight: $weight"
    python merge_by_addition.py --weight $weight --learn_which_modality music
done