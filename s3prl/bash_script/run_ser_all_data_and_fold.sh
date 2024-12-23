for upstream in hubert_large_ll60k; do 
for test_fold in fold1 fold2 fold3 fold4 fold5 ; do 

# All databases
if  [[ ${test_fold} == "fold1" ]]; then
    for corpus in CREMAD; do
    # The default config is "downstream/emotion/config.yaml"
    /home/ycevan/miniconda3/envs/s3prl/bin/python run_downstream.py -n SER_${upstream}_${corpus}_${test_fold}_5e-4_GR_lambda_4.0 -m train -u ${upstream} -d emotion_dro -c downstream/emotion_dev/config_${corpus}.yaml --start_saving_ckpt_step 2500 -o "config.downstream_expert.datarc.test_fold='$test_fold',,config.runner.total_steps=36000,,config.optimizer.lr=5e-4,,config.downstream_expert.debias.training_mode=GR,,config.downstream_expert.debias.lambda_GR=4"
    # python3 run_downstream.py -m evaluate -e result/downstream/${upstream}_${corpus}_$test_fold/dev-best.ckpt
    done;
else
    for corpus in CREMAD; do
    # The default config is "downstream/emotion/config.yaml"
    /home/ycevan/miniconda3/envs/s3prl/bin/python run_downstream.py -n SER_${upstream}_${corpus}_${test_fold}_5e-4_GR_lambda_4.0 -m train -u ${upstream} -d emotion_dro -c downstream/emotion_dev/config_${corpus}.yaml --start_saving_ckpt_step 2500 -o "config.downstream_expert.datarc.test_fold='$test_fold',,config.runner.total_steps=36000,,config.optimizer.lr=5e-4,,config.downstream_expert.debias.training_mode=GR,,config.downstream_expert.debias.lambda_GR=4"
    # python3 run_downstream.py -m evaluate -e result/downstream/${upstream}_${corpus}_$test_fold/dev-best.ckpt
    done;
fi
done;
done