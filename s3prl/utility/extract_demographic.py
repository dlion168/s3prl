import os
import re
import numpy as np
import argparse
import csv
from collections import defaultdict
from pathlib import Path

def read_log_and_average(exp_id, lr, base_path="downstream"):
    # 只匹配符合格式的資料夾名稱，例如 "{exp_id}_fold{i}_{lr}"
    folder_pattern = re.compile(rf"^{exp_id}_fold(\d+)_{lr}$")
    
    # 定義用來提取三類 bias 指標的正則表達式
    # 格式例如：
    # "test bias metrics on real gender: rms_tpr_gap=0.123, rms_fpr_gap=0.234, rms_f1_gap=0.345, rms_dp=0.456"
    pattern_gender = re.compile(
        r"^\w+\s+bias\s+metrics\s+on\s+real\s+gender:\s+rms_tpr_gap=([\d.]+),\s+rms_fpr_gap=([\d.]+),\s+rms_f1_gap=([\d.]+),\s+rms_dp=([\d.-]+)"
    )
    pattern_race = re.compile(
        r"^\w+\s+bias\s+metrics\s+on\s+race:\s+rms_tpr_gap=([\d.]+),\s+rms_fpr_gap=([\d.]+),\s+rms_f1_gap=([\d.]+),\s+rms_dp=([\d.-]+)"
    )
    pattern_agegroup = re.compile(
        r"^\w+\s+bias\s+metrics\s+on\s+agegroup:\s+rms_tpr_gap=([\d.]+),\s+rms_fpr_gap=([\d.]+),\s+rms_f1_gap=([\d.]+),\s+rms_dp=([\d.-]+)"
    )
    
    # 將各類 bias 指標的數值分別儲存在字典中（全域統計）
    metrics_per_fold = {
        "bias_gender": defaultdict(list),
        "bias_race": defaultdict(list),
        "bias_agegroup": defaultdict(list)
    }
    
    # 掃描 base_path 下的所有資料夾
    for folder_name in os.listdir(base_path):
        match = folder_pattern.match(folder_name)
        if not match:
            continue
        print(f"Processing folder: {folder_name}")
        log_path = os.path.join(base_path, folder_name, "log.log")
        if not os.path.exists(log_path):
            print(f"Log file not found for {folder_name}")
            continue

        # 初始化本檔案的旗標
        local_gender_found = False
        local_race_found = False
        local_age_found = False

        with open(log_path, "r") as log_file:
            lines = log_file.readlines()
        
        # 依次遍歷每一行，若符合格式則提取指標數值
        for line in lines:
            m = pattern_gender.match(line)
            if m:
                metrics_per_fold["bias_gender"]["rms_tpr_gap"].append(float(m.group(1)))
                metrics_per_fold["bias_gender"]["rms_fpr_gap"].append(float(m.group(2)))
                metrics_per_fold["bias_gender"]["rms_f1_gap"].append(float(m.group(3)))
                metrics_per_fold["bias_gender"]["rms_dp"].append(float(m.group(4)))
                local_gender_found = True
            m = pattern_race.match(line)
            if m:
                metrics_per_fold["bias_race"]["rms_tpr_gap"].append(float(m.group(1)))
                metrics_per_fold["bias_race"]["rms_fpr_gap"].append(float(m.group(2)))
                metrics_per_fold["bias_race"]["rms_f1_gap"].append(float(m.group(3)))
                metrics_per_fold["bias_race"]["rms_dp"].append(float(m.group(4)))
                local_race_found = True
            m = pattern_agegroup.match(line)
            if m:
                metrics_per_fold["bias_agegroup"]["rms_tpr_gap"].append(float(m.group(1)))
                metrics_per_fold["bias_agegroup"]["rms_fpr_gap"].append(float(m.group(2)))
                metrics_per_fold["bias_agegroup"]["rms_f1_gap"].append(float(m.group(3)))
                metrics_per_fold["bias_agegroup"]["rms_dp"].append(float(m.group(4)))
                local_age_found = True
        
        # 若某一項指標未在該 log 檔中找到，則列印錯誤訊息
        if not local_gender_found:
            print(f"Error: Folder {folder_name} does not contain bias metrics for real gender")
        if not local_race_found:
            print(f"Error: Folder {folder_name} does not contain bias metrics for race")
        if not local_age_found:
            print(f"Error: Folder {folder_name} does not contain bias metrics for agegroup")
    
    # 對每一類 bias 指標分別取平均
    averaged_metrics = {}
    if metrics_per_fold["bias_gender"]["rms_tpr_gap"]:
        avg_gender_tpr = np.mean(metrics_per_fold["bias_gender"]["rms_tpr_gap"])
        avg_gender_fpr = np.mean(metrics_per_fold["bias_gender"]["rms_fpr_gap"])
        avg_gender_f1  = np.mean(metrics_per_fold["bias_gender"]["rms_f1_gap"])
        avg_gender_dp  = np.mean(metrics_per_fold["bias_gender"]["rms_dp"])
    else:
        avg_gender_tpr = avg_gender_fpr = avg_gender_f1 = avg_gender_dp = None

    if metrics_per_fold["bias_race"]["rms_tpr_gap"]:
        avg_race_tpr = np.mean(metrics_per_fold["bias_race"]["rms_tpr_gap"])
        avg_race_fpr = np.mean(metrics_per_fold["bias_race"]["rms_fpr_gap"])
        avg_race_f1  = np.mean(metrics_per_fold["bias_race"]["rms_f1_gap"])
        avg_race_dp  = np.mean(metrics_per_fold["bias_race"]["rms_dp"])
    else:
        avg_race_tpr = avg_race_fpr = avg_race_f1 = avg_race_dp = None

    if metrics_per_fold["bias_agegroup"]["rms_tpr_gap"]:
        avg_age_tpr = np.mean(metrics_per_fold["bias_agegroup"]["rms_tpr_gap"])
        avg_age_fpr = np.mean(metrics_per_fold["bias_agegroup"]["rms_fpr_gap"])
        avg_age_f1  = np.mean(metrics_per_fold["bias_agegroup"]["rms_f1_gap"])
        avg_age_dp  = np.mean(metrics_per_fold["bias_agegroup"]["rms_dp"])
    else:
        avg_age_tpr = avg_age_fpr = avg_age_f1 = avg_age_dp = None

    # 將結果存入字典，以便後續 CSV 輸出
    averaged_metrics["gender TPR"] = avg_gender_tpr
    averaged_metrics["gender FPR"] = avg_gender_fpr
    averaged_metrics["gender F1"]  = avg_gender_f1
    averaged_metrics["gender DP"]  = avg_gender_dp
    averaged_metrics["race TPR"]   = avg_race_tpr
    averaged_metrics["race FPR"]   = avg_race_fpr
    averaged_metrics["race F1"]    = avg_race_f1
    averaged_metrics["race DP"]    = avg_race_dp
    averaged_metrics["age TPR"]    = avg_age_tpr
    averaged_metrics["age FPR"]    = avg_age_fpr
    averaged_metrics["age F1"]     = avg_age_f1
    averaged_metrics["age DP"]     = avg_age_dp

    return averaged_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate bias metrics from log files.")
    parser.add_argument("-e", "--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("-l", "--lr", type=str, required=True, help="Learning rate")
    parser.add_argument("-b", "--base_path", type=str, default="./result/downstream", help="Base path to the experiment folders")
    parser.add_argument("-o", "--output_csv", type=str, help="Path to save the output CSV file")
    args = parser.parse_args()

    results = read_log_and_average(args.exp_id, args.lr, args.base_path)

    # 定義 CSV 欄位：依照要求輸出 12 個欄位
    columns = [
        "gender TPR",
        "gender FPR",
        "gender F1",
        "gender DP",
        "race TPR",
        "race FPR",
        "race F1",
        "race DP",
        "age TPR",
        "age FPR",
        "age F1",
        "age DP"
    ]
    if args.output_csv is None:
        args.output_csv = f"./result/parsing/{args.exp_id}_{args.lr}_demo.csv"
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerow([results.get(col, None) for col in columns])

    print(f"Averaged bias metrics saved to {args.output_csv}")
