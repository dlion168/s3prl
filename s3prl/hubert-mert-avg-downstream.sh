#!/bin/bash
#SBATCH --job-name=hub-mert-avg
#SBATCH --cpus-per-task=24
#SBATCH --partition=RTXA6Kq --nodelist node09
#SBATCH --time=01-16:20:00
#SBATCH --output logfiles/downstream/hubert-mert-avg-teacher/vocalset_technique_id-task
##### BASELINE EXPERIMENT ######
source /export/home2/fabian/miniconda3/bin/activate s3prl_classic
BASE_DIR_S3PRL="/export/home2/fabian/projects/multi_distiller/s3prl/s3prl"
cd $BASE_DIR_S3PRL
stage="evaluating"
task=vocalset_technique_id #### for now I will only test KS and IC because of disk space limitation. add task : vocalset_singer_id    #### vocalset_singer_id   vocalset_technique_id
upstream="hubert_local"
echo "the upstream model is ! : ${exp_name}"
model="hubert-mert-teachers-average"
latest_checkpoint="result/pretrain/$model/averaged_model.ckpt" # ssast-huggingface-mert-distilhubert-all-from-ls960-init-from-hubert-averaged  mert-distilhubert-all-from-ls960-init-from-hubert-averaged
# Print the checkpoint being loaded
echo "Loading the latest model: $latest_checkpoint"
exp_setup=$model/$task
current_row=32

echo "##### running evaluation #####"
echo "$stage $task with model $distilled_model_checkpoint"
export CUDA_VISIBLE_DEVICES=7

module load cuda11.6/toolkit/11.6.0
export MPI_HOME="/cm/shared/apps/openmpi4/gcc/4.1.2/bin/mpicc"
export NCCL_HOME="/export/home2/fabian/toolkits/nccl"
export PATH=$NCCL_HOME/build/:/export/home2/fabian/bin/apr/bin:/export/home2/fabian/intel/oneapi/mkl/2023/bin:$PATH
export LD_LIBRARY_PATH=$NCCL_HOME/build/lib:/export/home2/fabian/bin/apr/lib:/export/home2/fabian/bin/subversion/lib:/export/home2/fabian/intel/oneapi/mkl/2023/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/build/include:/export/home2/fabian/bin/subversion/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$NCCL_HOME/build/include:/export/home2/fabian/bin/subversion/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX

downstream_path=/export/home2/fabian/projects/multi_distiller/s3prl/s3prl/result/downstream

if [ $task == "ic" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/fluent_commands/config.yaml" -u $upstream -k $latest_checkpoint -d fluent_commands -p ${downstream_path}/${exp_setup} --verbose --update_results --current_row-downstream $current_row 
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d fluent_commands \
          -c "./downstream/fluent_commands/config.yaml" \
          -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/intent_cl/fluent_speech_commands_dataset/fluent_speech_commands_dataset"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      echo "eval ic..."
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d fluent_commands \
          -c "./downstream/fluent_commands/config.yaml"
    fi
fi

if [ $task == "ks" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/speech_commands/config.yaml" -u $upstream -k $latest_checkpoint -d speech_commands -p ${downstream_path}/${exp_setup} --verbose \
      -o "config.downstream_expert.datarc.speech_commands_root=/export/home2/fabian/corpus/superb/keyword_spotting/speech_commands_v0.01,,\
          config.downstream_expert.datarc.speech_commands_test_root=/export/home2/fabian/corpus/superb/keyword_spotting/speech_commands_test_set_v0.01"
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d speech_commands \
          -c "./downstream/speech_commands/config.yaml" \
          -o "config.downstream_expert.datarc.speech_commands_root=/export/home2/fabian/corpus/superb/keyword_spotting/speech_commands_v0.01,,\
            config.downstream_expert.datarc.speech_commands_test_root=/export/home2/fabian/corpus/superb/keyword_spotting/speech_commands_test_set_v0.01"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d speech_commands \
          -c "./downstream/speech_commands/config.yaml"
    fi
fi

if [ $task == "vocalset_singer_id" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/vocalset_singer_id/config.yaml" -u $upstream -k $latest_checkpoint -d vocalset_singer_id -p ${downstream_path}/${exp_setup} --verbose \
      -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/vocalset"

      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d vocalset_singer_id \
          -c "./downstream/vocalset_singer_id/config.yaml" \
          -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/vocalset"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      echo "eval voca singer id..."
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d vocalset_singer_id \
          -c "./downstream/vocalset_singer_id/config.yaml" \
          -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/vocalset"
    fi
fi


if [ $task == "vocalset_technique_id" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/vocalset_technique_id/config.yaml" --update_results --current_row $current_row -u $upstream -k $latest_checkpoint -d vocalset_technique_id -p ${downstream_path}/${exp_setup} --verbose \
      -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/vocalset"

      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d vocalset_singer_id \
          -c "./downstream/vocalset_singer_id/config.yaml" \
          -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/vocalset"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      echo "eval voca singer id..."
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d vocalset_singer_id \
          -c "./downstream/vocalset_singer_id/config.yaml" \
          -o "config.downstream_expert.datarc.file_path=/export/home2/fabian/corpus/superb/vocalset"
    fi
fi



if [ $task == "asr" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL
 

    if [ $stage == "train" ]; then
      echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/asr/config.yaml" -u $upstream -k $latest_checkpoint -d asr -p ${downstream_path}/${exp_setup} --verbose \
        -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,\
        config.downstream_expert.datarc.libri_root=/export/home2/fabian/corpus/LibriSpeech,,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"

      
            echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-clean-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d asr -t "test-clean" \
          -c "./downstream/asr/config.yaml" \
          -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,\
              config.downstream_expert.datarc.libri_root=/export/home2/fabian/corpus/LibriSpeech,,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"



    elif [ $stage == "resuming" ]; then 
    echo "$stage $task"
    distributed="-m torch.distributed.launch --nproc_per_node ${gpus}";
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then
      echo "remember double check config files are working okay."
      echo "$stage $task"
     # Evaluation (evaluate finetune result on test dataset)"

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-clean-best.ckpt \
          -k $latest_checkpoint \
          -u $upstream --update_results --current_row $current_row \
          -d asr -t "test-clean" \
          -c "./downstream/asr/config.yaml" \
          -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,\
              config.downstream_expert.datarc.libri_root=/export/home2/fabian/corpus/LibriSpeech,,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"


    fi

fi
