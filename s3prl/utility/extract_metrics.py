import os
import re
import numpy as np
import argparse
import csv
from collections import defaultdict
from pathlib import Path

def read_log_and_average(exp_id, lr, base_path="downstream"):
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

        with open(log_path, "r") as log_file:
            lines = log_file.readlines()

        # Find the last best step
        best_step = None
        for line in lines:
            best_match = re.search(r"New best on dev loss at step (\d+): ([\d.]+)", line)
            if best_match:
                best_step = int(best_match.group(1))

        if best_step is None:
            print(f"No best step found in {log_path}")
            continue

        # After we find best_step, we consider lines after that for metrics
        best_step_index = None
        for i, line in enumerate(lines):
            if f"New best on dev loss at step {best_step}:" in line:
                best_step_index = i
        if best_step_index is None:
            print(f"No line found for the best step {best_step} in {log_path}")
            continue

        # We'll keep track of which RMS+max metrics we have found to avoid duplicates
        found_rms_max = set()

        # Parse lines after best_step for single and RMS+max metrics
        for line in lines[best_step_index+1:]:
            # If we hit another "New best..." line, stop
            if "New best on dev loss at step" in line:
                break

            # Parse single metrics with "at step {best_step}"
            for m in single_metrics:
                if m in line and f"at step {best_step}:" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        value_str = parts[-1].strip()
                        try:
                            value = float(value_str)
                            metrics_per_fold[m].append(value)
                        except ValueError:
                            pass

            # Parse RMS and max metrics
            # Example line for RMS+max:
            # "test DP RMS disparity: 0.02396, max disparity: 0.05862"
            # We only record the first occurrence per fold for these metrics
            for (rms_key, max_key, max_label) in rms_max_pairs:
                if (rms_key in columns) and (max_key in columns):
                    # Check if we already found them for this fold
                    if rms_key in found_rms_max or max_key in found_rms_max:
                        continue

                    # Line starts with something like "test DP RMS disparity"
                    if line.startswith(rms_key):
                        # Extract two floats
                        # Format is something like:
                        # test DP RMS disparity: {rms_val}, max disparity: {max_val}
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
    args = parser.parse_args()

    results = read_log_and_average(args.exp_id, args.lr, args.base_path)

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
