import os
import math
import torch
import random
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

import json
import numpy as np
import warnings
import pickle as pk
from sklearn.metrics import classification_report
from collections import defaultdict

from .dataset import prepare_datasets, collate_fn_padd, WeightedDataset
from ..model import *

warnings.filterwarnings("ignore")

def class_balanced_softmax_cross_entropy_with_softtarget(logits, targets, weights, reduction='mean'):
    weights = weights.unsqueeze(0).repeat(targets.shape[0], 1) * targets
    weights = weights.sum(dim=1, keepdim=True).repeat(1, targets.shape[1])
    log_probs = F.log_softmax(logits, dim=1)
    batch_loss = -torch.sum(weights * targets * log_probs, dim=1)
    if reduction == 'none':
        return batch_loss
    elif reduction == 'mean':
        return torch.mean(batch_loss)
    elif reduction == 'sum':
        return torch.sum(batch_loss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

def safe_div(a, b):
    return a / b if b > 0 else 0.0
            
def rms_gap(values):
    values = np.array(values)
    if len(values) == 0:
        return 0.0
    mean_val = values.mean()
    diff = values - mean_val
    return math.sqrt(np.mean(diff**2))
class DownstreamExpert(nn.Module):
    """
    training_mode (merged):
    - "ERM": Just ERM
    - "GroupDRO": Group Distributionally Robust Optimization
    - "DS": Downsampling debiasing
    - "RW": Reweighting debiasing
    - "GR": Gap Regularization debiasing using TPR difference and FPR difference
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.training_mode = downstream_expert['debias'].get('training_mode', 'ERM')  
        # Possible values: "ERM", "GroupDRO", "DS", "RW", "GR"
        self.lambda_GR = downstream_expert['debias'].get('lambda_GR', 1.0)

        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        self.audio_path = os.path.join(self.datarc['root'], self.datarc['corpus'], "Audios")
        self.labels_path = os.path.join(self.datarc['root'], self.datarc['corpus'],
                                        self.datarc['p_or_s'], 
                                        "labels_consensus_" + self.datarc['test_fold'].replace("fold", "") + ".csv")
        self.config_path = os.path.join(self.datarc['root'], self.datarc['corpus'], self.datarc['p_or_s'], "config.json")

        (self.train_dataset, 
         self.dev_dataset, 
         self.test_dataset, 
         self.class_balanced_weights, 
         self.k_thresold, 
         self.all_emotions) = prepare_datasets(self.datarc, self.config_path)
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=len(self.config['categorical']["emo_type"]),
            **model_conf,
        )

        self.objective = class_balanced_softmax_cross_entropy_with_softtarget
        self.expdir = expdir
        self.register_buffer('best_score', torch.ones(1)*99999)

        # Apply debiasing if chosen
        self.apply_debiasing(self.training_mode)

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')

    def apply_debiasing(self, method):
        """
        For DS (downsampling), we subset the dataset.
        For RW (reweighting), we wrap the dataset with WeightedDataset using computed weights.
        For ERM/GroupDRO/GR, we also wrap with WeightedDataset, but assign equal weights=1.
        """
        # Collect group info
        class_gender_pairs = []
        for idx in range(len(self.train_dataset)):
            wav, lab, utt, gender = self.train_dataset[idx]
            class_idx = np.argmax(lab)
            class_gender_pairs.append((class_idx, gender, idx))

        counts = defaultdict(int)
        class_counts = defaultdict(int)
        # Group samples by class and gender
        class_genders = defaultdict(lambda: defaultdict(list))
        for (c, g, i) in class_gender_pairs:
            counts[(c,g)] += 1
            class_counts[c] += 1
            class_genders[c][g].append(i)

        if method == "DS":
            # Downsampling
            if len(class_gender_pairs) == 0:
                # If no counts (empty), just assign equal weights to all instances
                weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
                self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
                return

            new_indices = []
            # For each class c, find the minimum count across all genders and downsample accordingly
            for c, gender_dict in class_genders.items():
                # Find the minimum count for this class across all genders
                min_count_class = min(len(idx_list) for idx_list in gender_dict.values())
                # Downsample each gender of this class to min_count_class
                for g, idx_list in gender_dict.items():
                    random.shuffle(idx_list)
                    chosen = idx_list[:min_count_class]
                    new_indices.extend(chosen)

            self.train_dataset = Subset(self.train_dataset, new_indices)
            # After downsampling, assign uniform weights=1
            weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
            self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
            print(f"[DS] Downsampled training set to {len(new_indices)} samples.")

        elif method == "RW":
            # Reweighting
            if len(counts) == 0:
                # No groups, uniform weights
                weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
            else:
                weight_dict = {}
                # For each class c, find how many gender categories
                # and total instances of class c is class_counts[c]
                # G_c = number of genders for class c
                for c, gender_dict in class_genders.items():
                    G_c = len(gender_dict)            # number_of_genders_for_c
                    sum_c = class_counts[c]           # total instances with class c

                    for g, idx_list in gender_dict.items():
                        group_count = counts[(c,g)]
                        # w = (class_counts[c]/G_c) * (1.0 / counts[(c,g)])
                        w = (sum_c / G_c) * (1.0 / group_count)
                        for idx_sample in idx_list:
                            weight_dict[idx_sample] = w
                self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
            print("[RW] Assigned reweighting to training samples.")
        else:
            # ERM or GroupDRO: just assign uniform weights=1
            weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
            self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
        self.dev_dataset = WeightedDataset(self.dev_dataset, weight_dict)
        self.test_dataset = WeightedDataset(self.test_dataset, weight_dict)

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn_padd
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc['eval_batch_size'],
            shuffle=False,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn_padd
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode):
        return getattr(self, f'get_{mode}_dataloader')()

    def forward(self, mode, features, labels, filenames, gender_labels, sample_weights, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device)
        padded_features = pad_sequence(features, batch_first=True).to(device)
        projected_features = self.projector(padded_features)
        predicted_logits, hidden_states = self.model(projected_features, features_len)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)

        per_sample_loss = self.objective(predicted_logits, labels, self.class_balanced_weights.to(device), reduction='none')

        # Apply weights directly (uniform=1 for non-RW, or actual weights for RW)
        per_sample_loss = per_sample_loss * sample_weights

        if self.training_mode in ["ERM", "DS", "RW"]:
            total_loss = per_sample_loss.mean()
        elif self.training_mode == "GroupDRO":
            gender_labels = gender_labels.to(device)
            unique_genders = torch.unique(gender_labels)
            loss_g_list = []
            for g in unique_genders:
                mask = (gender_labels == g)
                if mask.sum() > 0:
                    group_loss = per_sample_loss[mask].mean()
                    loss_g_list.append((group_loss, g))
            if len(loss_g_list) == 0:
                total_loss = per_sample_loss.mean()
            else:
                worst_L_g, worst_g = max(loss_g_list, key=lambda x: x[0])
                total_loss = worst_L_g
        elif self.training_mode == "GR":
            # predictions_binary, labels_binary, gender_labels are already torch tensors
            # predictions_binary: (B, C)
            # labels_binary: (B, C)
            # gender_labels: (B,)

            gender_labels = gender_labels.to(device)
            unique_genders = torch.unique(gender_labels)

            TPR_diffs = []
            FPR_diffs = []

            # Some small epsilon to avoid division by zero
            eps = 1e-8

            B, C = labels_binary.shape
            for c in range(C):
                pred_c = predictions_binary[:, c]   # (B,)
                label_c = labels_binary[:, c]       # (B,)

                # Compute per-gender TPR & FPR
                gender_TPR = {}
                gender_FPR = {}

                for g in unique_genders:
                    mask = (gender_labels == g).float()  # (B,)
                    TP_g = (pred_c * label_c * mask).sum()
                    FN_g = ((1 - pred_c) * label_c * mask).sum()
                    FP_g = (pred_c * (1 - label_c) * mask).sum()
                    TN_g = ((1 - pred_c) * (1 - label_c) * mask).sum()

                    # Compute TPR and FPR
                    TPR_g = TP_g / (TP_g + FN_g + eps)
                    FPR_g = FP_g / (FP_g + TN_g + eps)

                    gender_TPR[g.item()] = TPR_g
                    gender_FPR[g.item()] = FPR_g

                # Compute pairwise differences for TPR and FPR
                g_list = list(gender_TPR.keys())
                for i in range(len(g_list)):
                    for j in range(i+1, len(g_list)):
                        g1, g2 = g_list[i], g_list[j]
                        # Differences are still tensors, so gradients can flow
                        TPR_diff = (gender_TPR[g1] - gender_TPR[g2]).abs()
                        FPR_diff = (gender_FPR[g1] - gender_FPR[g2]).abs()
                        TPR_diffs.append(TPR_diff)
                        FPR_diffs.append(FPR_diff)

            # Convert lists to tensors if they are not empty; if empty, set them to zero
            if len(TPR_diffs) == 0:
                TPR_RMS_gap = torch.tensor(0.0, device=device)
            else:
                TPR_diffs_tensor = torch.stack(TPR_diffs)
                mean_TPR = TPR_diffs_tensor.mean()
                TPR_RMS_gap = torch.sqrt(((TPR_diffs_tensor - mean_TPR)**2).mean())

            if len(FPR_diffs) == 0:
                FPR_RMS_gap = torch.tensor(0.0, device=device)
            else:
                FPR_diffs_tensor = torch.stack(FPR_diffs)
                mean_FPR = FPR_diffs_tensor.mean()
                FPR_RMS_gap = torch.sqrt(((FPR_diffs_tensor - mean_FPR)**2).mean())

            classification_loss = per_sample_loss.mean()
            total_loss = classification_loss + self.lambda_GR * (TPR_RMS_gap + FPR_RMS_gap)
        else:
            raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

        # Compute binary predictions
        prediction_distribution = F.softmax(predicted_logits, dim=1)
        predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
        labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

        if "all_predictions_binary" not in records:
            records["all_predictions_binary"] = []
            records["all_labels_binary"] = []
            records["all_genders"] = []
            records["filename"] = []
            records["predict"] = []
            records["truth"] = []
            records["loss"] = []

        records["all_predictions_binary"].append(predictions_binary.cpu().numpy())
        records["all_labels_binary"].append(labels_binary.cpu().numpy())
        if gender_labels is not None:
            records["all_genders"].append(gender_labels.cpu().numpy())
        else:
            records["all_genders"].append(np.zeros((len(labels_binary),), dtype=np.int64))

        # Store loss for later averaging
        records["loss"].append(total_loss.item())
        records["filename"] += filenames

        # Convert predictions back to emo strings for logging
        all_emotions_np = np.array(self.all_emotions)
        for idx in range(len(labels_binary)):
            true_emo = ";".join(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True)==1.0)[0]])
            pred_emo = ";".join(all_emotions_np[np.where(predictions_binary[idx].cpu().numpy(force=True)==1.0)[0]])
            records["truth"].append(true_emo)
            records["predict"].append(pred_emo)

        return total_loss

    def log_records(self, mode, records, logger, global_step, **kwargs):
        # Compute macro-f1 and acc here
        all_preds = np.concatenate(records["all_predictions_binary"], axis=0)  # (N, C)
        all_labels = np.concatenate(records["all_labels_binary"], axis=0)       # (N, C)

        # macro-f1 from classification_report
        reprot_dict = classification_report(all_labels, all_preds, target_names=self.all_emotions, output_dict=True)
        macro_f1 = reprot_dict['macro avg']['f1-score']

        # acc by one-vs-all accuracy
        N, C = all_labels.shape
        acc_list = []
        for c in range(C):
            pred_c = all_preds[:, c]
            label_c = all_labels[:, c]
            TP = np.sum((pred_c == 1) & (label_c == 1))
            FP = np.sum((pred_c == 1) & (label_c == 0))
            FN = np.sum((pred_c == 0) & (label_c == 1))
            TN = np.sum((pred_c == 0) & (label_c == 0))
            denom = (TP + TN + FP + FN)
            accuracy_c = (TP + TN) / denom if denom > 0 else 0
            acc_list.append(accuracy_c)
        acc = np.mean(acc_list) if len(acc_list) > 0 else 0.0

        # Average loss
        average_loss = torch.FloatTensor(records['loss']).mean().item()

        # Log macro-f1, loss, acc
        metrics_to_log = {
            'macro-f1': macro_f1,
            'acc': acc,
            'loss': average_loss
        }

        save_names = []
        for key, val in metrics_to_log.items():
            logger.add_scalar(f'emotion-{self.fold}/{mode}-{key}', val, global_step=global_step)
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} {key}: {val}")
                f.write(f'{mode} {key} at step {global_step}: {val}\n')
            if key == 'loss' and mode == 'dev' and val < self.best_score:
                self.best_score = torch.ones(1)*val
                with open(Path(self.expdir) / "log.log", 'a') as f:
                    f.write(f'New best on {mode} {key} at step {global_step}: {val}\n')
                save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            all_genders = np.concatenate(records["all_genders"], axis=0)  # (N,)
            unique_genders = np.unique(all_genders)

            # To store differences for each metric across genders and classes
            TPR_diffs = []
            FPR_diffs = []
            F1_diffs = []

            # Compute TPR, FPR, f1 per class per gender
            # We'll have a structure: For each class c:
            #   For each gender g in unique_genders: compute metrics and store
            for c in range(C):
                pred_c = all_preds[:, c]
                label_c = all_labels[:, c]

                # Compute per-gender metrics
                gender_metrics = {}
                for g in unique_genders:
                    mask = (all_genders == g)
                    pred_g = pred_c[mask]
                    label_g = label_c[mask]

                    TP_g = np.sum((pred_g == 1) & (label_g == 1))
                    FP_g = np.sum((pred_g == 1) & (label_g == 0))
                    FN_g = np.sum((pred_g == 0) & (label_g == 1))
                    TN_g = np.sum((pred_g == 0) & (label_g == 0))

                    TPR_g = safe_div(TP_g, TP_g+FN_g)
                    FPR_g = safe_div(FP_g, FP_g+TN_g)
                    precision_g = safe_div(TP_g, TP_g+FP_g)
                    recall_g = TPR_g
                    f1_g = safe_div(2*precision_g*recall_g, precision_g+recall_g) if (precision_g+recall_g)>0 else 0.0

                    gender_metrics[g] = (TPR_g, FPR_g, f1_g)

                # Now compute differences between each pair of genders for this class
                # If only two genders, it's straightforward; if more, do pairwise
                g_list = list(gender_metrics.keys())
                for i in range(len(g_list)):
                    for j in range(i+1, len(g_list)):
                        g1, g2 = g_list[i], g_list[j]
                        TPR_diff = abs(gender_metrics[g1][0] - gender_metrics[g2][0])
                        FPR_diff = abs(gender_metrics[g1][1] - gender_metrics[g2][1])
                        F1_diff = abs(gender_metrics[g1][2] - gender_metrics[g2][2])

                        TPR_diffs.append(TPR_diff)
                        FPR_diffs.append(FPR_diff)
                        F1_diffs.append(F1_diff)

            # Compute RMS and max for each metric
            rms_tpr_gap = rms_gap(TPR_diffs)
            rms_fpr_gap = rms_gap(FPR_diffs)
            rms_f1_gap = rms_gap(F1_diffs)

            max_tpr_gap = max(TPR_diffs) if len(TPR_diffs)>0 else 0.0
            max_fpr_gap = max(FPR_diffs) if len(FPR_diffs)>0 else 0.0
            max_f1_gap = max(F1_diffs) if len(F1_diffs)>0 else 0.0

            DP_disparities = []
            for c in range(C):
                pred_c = all_preds[:, c]
                label_c = all_labels[:, c]
                pred_1 = np.sum(pred_c==1)
                n_c = len(pred_c)
                global_pos = safe_div(pred_1, n_c)
                dp_vals = []
                for z in unique_genders:
                    z_mask = (all_genders == z)
                    pred_1_z = np.sum(pred_c[z_mask]==1)
                    n_z = np.sum(z_mask)
                    p_yhat_1_z = safe_div(pred_1_z, n_z)
                    dp_vals.append(abs(p_yhat_1_z - global_pos))
                if len(dp_vals)>0:
                    DP_disparities.append(np.max(dp_vals))
                else:
                    DP_disparities.append(0.0)

            rms_dp = rms_gap(DP_disparities)
            max_dp_gap = np.max(DP_disparities) if len(DP_disparities)>0 else 0.0

            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} TPR RMS gap: {rms_tpr_gap}, max gap: {max_tpr_gap}")
                f.write(f"{mode} TPR RMS gap: {rms_tpr_gap}, max gap: {max_tpr_gap}\n")
                print(f"{mode} FPR RMS gap: {rms_fpr_gap}, max gap: {max_fpr_gap}")
                f.write(f"{mode} FPR RMS gap: {rms_fpr_gap}, max gap: {max_fpr_gap}\n")
                print(f"{mode} F1 RMS gap: {rms_f1_gap}, max gap: {max_f1_gap}")
                f.write(f"{mode} F1 RMS gap: {rms_f1_gap}, max gap: {max_f1_gap}\n")
                print(f"{mode} DP RMS disparity: {rms_dp}, max disparity: {max_dp_gap}")
                f.write(f"{mode} DP RMS disparity: {rms_dp}, max disparity: {max_dp_gap}\n")

            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fname} {pred}\n" for fname, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fname} {tr}\n" for fname, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names

