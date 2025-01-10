import os
import re
import numpy as np
import argparse
import csv
from collections import defaultdict
from pathlib import Path

def read_log_and_average(exp_id, lr, base_path="downstream", metric="loss", start_from_step=0):
    # Pattern to match folders like "{exp_id}_fold{i}_{lr}"
    pattern = re.compile(rf"^{exp_id}_fold(\d+)_{lr}$")

    # Single-value test metrics that appear with "at step {best_step}"
    single_metrics = [
        "test macro-f1",
        "test acc",
        "test loss"
    ]

    # Metrics that appear as RMS and max on the same line (without "at step"):
    # We'll parse them after the best step line, first occurrence only.
    rms_max_pairs = [
        ("test TPR RMS gap", "test TPR max gap", "max gap"),
        ("test FPR RMS gap", "test FPR max gap", "max gap"),
        ("test F1 RMS gap", "test F1 max gap", "max gap"),
        ("test DP RMS disparity", "test DP max disparity", "max disparity")
    ]

    # Define final columns (including RMS and max)
    columns = [
        "test macro-f1",
        "test acc",
        "test loss",
        "test TPR RMS gap",
        "test TPR max gap",
        "test FPR RMS gap",
        "test FPR max gap",
        "test F1 RMS gap",
        "test F1 max gap",
        "test DP RMS disparity",
        "test DP max disparity"
    ]

    # Store metrics per fold
    metrics_per_fold = defaultdict(list)

    for folder_name in os.listdir(base_path):
        match = pattern.match(folder_name)
        if not match:
            continue
        print(folder_name)
        fold = int(match.group(1))
        log_path = os.path.join(base_path, folder_name, "log.log")

        if not os.path.exists(log_path):
            print(f"Log file not found for {folder_name}")
            continue

        # ---------------------------------------------------------
        # (1) 首先掃描整個 log 檔，找到 dev GR_loss 最小的那個 step
        #     這個 step 視為 best_step (我們要記錄該 checkpoint 的指標)
        # ---------------------------------------------------------
        lines = []
        with open(log_path, "r") as log_file:
            lines = log_file.readlines()
        
        best_step = None
        best_gr_loss = float("inf")

        # 找出最小的 dev GR_loss, 其行格式如：
        # dev GR_loss at step 4653: 1.036889672279358
        # 依此找出整個檔案的最小值
        for line in lines:
            gr_loss_match = re.search(fr"dev {metric} at step (\d+): ([\d.]+)", line)
            if gr_loss_match:
                step_num = int(gr_loss_match.group(1))
                if step_num >= start_from_step: 
                    gr_loss_val = float(gr_loss_match.group(2))
                    if gr_loss_val < best_gr_loss:
                        best_gr_loss = gr_loss_val
                        best_step = step_num

        if best_step is None:
            print(f"No dev {metric} found in {log_path}, skip this fold.")
            continue
        else:
            print(f"Found best_step={best_step} (smallest dev {metric}={best_gr_loss}) for fold {fold}.")

        # ---------------------------------------------------------
        # (2) 根據找到的 best_step，再去抓取該 step 後續 (或同行) 的相關測試指標
        #     這裡邏輯跟原本相同：parse single metrics 與 RMS+max pairs
        # ---------------------------------------------------------
        # 先找出 "dev GR_loss at step best_step:" 行的索引
        best_step_index = None
        pattern_for_best = f"dev {metric} at step {best_step}:"
        for i, line in enumerate(lines):
            if pattern_for_best in line:
                best_step_index = i
                break

        if best_step_index is None:
            print(f"No line found for dev {metric} at step {best_step} in {log_path}")
            continue

        # 我們用 set 來避免重複擷取 RMS+max 這些 pair
        found_rms_max = set()

        # 從 best_step_index 之後，往下 parse 直到出現下一個 step 的紀錄或檔案結尾
        # 停止條件：若發現類似 " at step X:" ，且 X != best_step，就停止。
        # 但若你的 log 檔不會這樣中斷，也可自行調整停止判斷。
        for line in lines[best_step_index+1:]:
            # 如果偵測到另一個 step 的紀錄就停止
            # 例如 "train macro-f1 at step 4935:" 或 "dev GR_loss at step 4935:"
            # 只要 step != best_step，就代表我們超過了這個 checkpoint 的區塊
            next_step_match = re.search(r"at step (\d+):", line)
            if next_step_match:
                step_num = int(next_step_match.group(1))
                if step_num != best_step:
                    break

            # Parse single metrics with "at step {best_step}"
            for m in single_metrics:
                # e.g., "test macro-f1 at step 4653: 0.6351606678128819"
                if m in line and f"at step {best_step}:" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        value_str = parts[-1].strip()
                        try:
                            value = float(value_str)
                            metrics_per_fold[m].append(value)
                        except ValueError:
                            pass

            # Parse RMS + max metrics
            # 例如：
            # test DP RMS disparity: 0.02396, max disparity: 0.05862
            for (rms_key, max_key, max_label) in rms_max_pairs:
                if (rms_key in columns) and (max_key in columns):
                    if rms_key in found_rms_max or max_key in found_rms_max:
                        continue
                    if line.startswith(rms_key):
                        m = re.search(r": ([\d.]+), max .*: ([\d.]+)", line)
                        if m:
                            rms_val = float(m.group(1))
                            max_val = float(m.group(2))
                            metrics_per_fold[rms_key].append(rms_val)
                            metrics_per_fold[max_key].append(max_val)
                            found_rms_max.add(rms_key)
                            found_rms_max.add(max_key)

    # Average metrics
    averaged_metrics = {}
    for key in columns:
        averaged_metrics[key] = np.mean(metrics_per_fold[key]) if metrics_per_fold[key] else None

    return averaged_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metrics from log files.")
    parser.add_argument("-e", "--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("-l", "--lr", type=str, required=True, help="Learning rate")
    parser.add_argument("-b", "--base_path", type=str, default="./result/downstream", help="Base path to the experiment folders")
    parser.add_argument("-o", "--output_csv", type=str, help="Path to save the output CSV file")
    parser.add_argument("-m", "--metric", type=str, help="Lowest metric to parse from", default="loss")
    parser.add_argument("-s", "--start_from_step", type=int, help="Parsing metrics after this step", default=0)
    args = parser.parse_args()

    results = read_log_and_average(args.exp_id, args.lr, args.base_path, args.metric, args.start_from_step)

    columns = [
        "test macro-f1",
        "test acc",
        "test TPR RMS gap",
        "test TPR max gap",
        "test FPR RMS gap",
        "test FPR max gap",
        "test F1 RMS gap",
        "test F1 max gap",
        "test DP RMS disparity",
        "test DP max disparity"
    ]
    if args.output_csv == None:
        args.output_csv = f"./result/parsing/{args.exp_id}_{args.lr}.csv"
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerow([results.get(col, None) for col in columns])

    print(f"Averaged metrics saved to {args.output_csv}")
