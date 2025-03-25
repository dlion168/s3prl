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
    """
    依照 class_balanced + softtarget 設計的 cross entropy:
      1. 將 class_weights 依照 targets (one-hot/soft) 進行逐樣本加權
      2. 將加權後的目標分佈與 logits 做 cross entropy
    輸入:
      logits: (B, C)
      targets: (B, C)
      weights: (C,)  -> 各 class 的權重
    輸出:
      batch_loss: shape = (B,) or scalar (取決於 reduction)
    """
    expanded_w = weights.unsqueeze(0).repeat(targets.shape[0], 1)  # (B, C)
    per_sample_class_weights = (expanded_w * targets).sum(dim=1, keepdim=True)  # (B,1)
    broadcast_w = per_sample_class_weights.repeat(1, targets.shape[1])          # (B,C)

    log_probs = F.log_softmax(logits, dim=1)                                   # (B,C)
    batch_loss = -torch.sum(broadcast_w * targets * log_probs, dim=1)          # (B,)

    if reduction == 'none':
        return batch_loss
    elif reduction == 'mean':
        return torch.mean(batch_loss)
    elif reduction == 'sum':
        return torch.sum(batch_loss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')


def class_balanced_generalized_cross_entropy_loss(logits, targets, weights, q=0.7, reduction='mean'):
    """
    依照 LfF 的 Generalized Cross Entropy (GCE) 設計，但增添 class_balanced，
    GCE(p, y) = \sum_{i=1}^{C} [ w_i * y_i * (1 - p_i^q)/q ] 
    其中:
    - p_i = softmax(logits)[..., i]
    - w_i = weights[i]
    - y_i = targets[..., i]
    """
    prob = F.softmax(logits, dim=1)              # (B, C)
    expanded_w = weights.unsqueeze(0).repeat(targets.shape[0], 1)  # (B, C)
    weighted_targets = expanded_w * targets      # (B, C)

    one_minus_prob_q = (1.0 - prob.pow(q))       # (B, C)
    gce_each_class = weighted_targets * one_minus_prob_q / q  # (B, C)
    loss_per_sample = torch.sum(gce_each_class, dim=1)        # (B,)

    if reduction == 'none':
        return loss_per_sample
    elif reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:
        raise NotImplementedError('Unsupported reduction mode.')


def safe_div(a, b):
    return a / b if b > 0 else 0.0

def rms_gap(values):
    values = np.array(values)
    if len(values) == 0:
        return 0.0
    return math.sqrt(np.mean(values**2))

class DownstreamExpert(nn.Module):
    """
    training_mode (merged):
    - "ERM": Just ERM
    - "GroupDRO": Group Distributionally Robust Optimization
    - "DS": Downsampling debiasing
    - "RW": Reweighting debiasing
    - "GR": Gap Regularization debiasing
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.training_mode = downstream_expert['debias'].get('training_mode', 'ERM')

        self.lambda_GR = downstream_expert['debias'].get('lambda_GR', 1.0)
        self.lambda_GDRO = downstream_expert['debias'].get('lambda_GDRO', 0.0)
        
        # BPA 模式的 EMA 參數
        self.bpa_ema_alpha = downstream_expert['debias'].get('bpa_ema_alpha', 0.3)

        # 用來記錄 (cluster -> importance) 的動態加權
        self.cluster_importance = defaultdict(float)
        
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        self.audio_path = os.path.join(self.datarc['root'], self.datarc['corpus'], "Audios")
        self.config_path = os.path.join(self.datarc['root'], self.datarc['corpus'], self.datarc['p_or_s'], "config.json")

        (self.train_dataset, 
         self.dev_dataset, 
         self.test_dataset, 
         self.class_balanced_weights, 
         self.k_thresold, 
         self.all_emotions) = prepare_datasets(self.datarc, self.config_path)
        
        # 讀取 config.json 中的 GR_target 設定
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        self.GR_target = self.config.get('GR_target', 'TPR+FPR')  
        # 預設為 'TPR+FPR'，可在 config.json 裡設定為 "TPR" 或 "FPR"

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        # ---------------------- Debiased Model ----------------------
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = len(self.config['categorical']["emo_type"]),
            **model_conf,
        )

        # --------------------- Biased Model (for LfF) ---------------------

        # LfF 執行時，我們會在 forward() 中用到的 EMA buffer (更新於每個 batch)
        self.ema_loss_b = []
        self.ema_loss_d = []

        # 給其他模式 (ERM / RW / DS / GroupDRO / GR) 用的損失函式
        self.objective = class_balanced_softmax_cross_entropy_with_softtarget

        self.expdir = expdir
        self.register_buffer('best_score', torch.ones(1)*99999)

        # Apply debiasing if chosen
        self.apply_debiasing(self.training_mode)
        self.start_saving_ckpt_step = kwargs['start_saving_ckpt_step']

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')

    def apply_debiasing(self, method):
        """
        根據 debiasing 方法：
          - DS: Downsampling（根據 pseudo group 下採樣）
          - RW: Reweighting（根據 pseudo group 重新加權）
          - ERM/GroupDRO/GR/LfF: 均設 uniform weight = 1
        """
        # 收集分群資訊：使用 pseudo group 僅採用 K_Means（gender_labels[1]）
        class_gender_pairs = []
        for idx in range(len(self.train_dataset)):
            wav, lab, utt, gender = self.train_dataset[idx]
            class_idx = np.argmax(lab)
            pseudo_group = int(round(gender[1]))  # 僅使用 K_Means 作為 pseudo group
            class_gender_pairs.append((class_idx, pseudo_group, idx))
        
        dev_class_gender_pairs = []
        for idx in range(len(self.dev_dataset)):
            wav, lab, utt, gender = self.dev_dataset[idx]
            class_idx = np.argmax(lab)
            pseudo_group = int(round(gender[1]))
            dev_class_gender_pairs.append((class_idx, pseudo_group, idx))

        counts = defaultdict(int)
        class_counts = defaultdict(int)
        class_genders = defaultdict(lambda: defaultdict(list))
        dev_class_genders = defaultdict(lambda: defaultdict(list))
        self.gender_count = defaultdict(int)
        
        for (c, g, i) in class_gender_pairs:
            counts[(c, g)] += 1
            class_counts[c] += 1
            self.gender_count[g] += 1
            class_genders[c][g].append(i)

        for (c, g, i) in dev_class_gender_pairs:
            dev_class_genders[c][g].append(i)
        
        if method == "DS":
            # Downsampling
            if len(class_gender_pairs) == 0:
                weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
                self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
                return

            new_indices = []
            for c, gender_dict in class_genders.items():
                min_count_class = min(len(idx_list) for idx_list in gender_dict.values())
                for g, idx_list in gender_dict.items():
                    random.shuffle(idx_list)
                    chosen = idx_list[:min_count_class]
                    new_indices.extend(chosen)

            self.train_dataset = Subset(self.train_dataset, new_indices)
            weight_dict = {i:1.0 for i in range(max(len(self.train_dataset), len(self.dev_dataset), len(self.test_dataset)))}
            self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
            self.dev_dataset = WeightedDataset(self.dev_dataset, weight_dict)
            print(f"[DS] Downsampled training set to {len(new_indices)} samples.")

        elif method == "RW":
            # Reweighting
            if len(counts) == 0:
                weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
                dev_weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
            else:
                weight_dict = {}
                dev_weight_dict = {}
                for c, gender_dict in class_genders.items():
                    G_c = len(gender_dict)
                    sum_c = class_counts[c]
                    for g, idx_list in gender_dict.items():
                        group_count = counts[(c, g)]
                        w = (sum_c / G_c) * (1.0 / group_count)
                        for idx_sample in idx_list:
                            weight_dict[idx_sample] = w
                        for dev_idx_sample in dev_class_genders[c][g]:
                            dev_weight_dict[dev_idx_sample] = w
                
                # 對於 dev 中存在但 train 中缺少的組別，賦予 weight 1
                for c_dev, gender_dict_dev in dev_class_genders.items():
                    for g, idx_list in gender_dict_dev.items():
                        if g not in class_genders[c_dev]:
                            for idx_sample in idx_list:
                                dev_weight_dict[idx_sample] = 1.0

                self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
                self.dev_dataset = WeightedDataset(self.dev_dataset, dev_weight_dict)
            print("[RW] Assigned reweighting to training samples.")

        else:
            # ERM, GroupDRO, GR, LfF: 統一權重 1
            weight_dict = {i:1.0 for i in range(len(self.train_dataset))}
            self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
            self.dev_dataset = WeightedDataset(self.dev_dataset, weight_dict)
        self.test_dataset = WeightedDataset(self.test_dataset, {i:1.0 for i in range(len(self.test_dataset))})

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
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)
        projected_features = self.projector(padded_features)
        logits_debiased, _ = self.model(projected_features, features_len)
        
        # 根據新的 gender_labels 結構：[Gender, K_Means, Race, AgeGroup]
        # 若為 test 模式，debiasing 時仍採用真實性別 (第一個元素)；否則 debiasing 使用 K_Means (第二個元素)
        if mode == "test":
            gender_hard_labels = gender_labels[:, 0]
        else:
            gender_hard_labels = gender_labels[:, 1].clone().detach().to(torch.int)
        
        # 除 debiasing 使用外，另外紀錄真實性別、種族與年齡群供 bias 評估用
        real_gender = gender_labels[:, 0]
        race = gender_labels[:, 2]
        agegroup = gender_labels[:, 3]
        if "all_real_gender" not in records: 
            records["all_real_gender"] = []
        if "all_race" not in records:
            records["all_race"] = []
        if "all_agegroup" not in records:
            records["all_agegroup"] = []
        records["all_real_gender"].append(real_gender.cpu().numpy())
        records["all_race"].append(race.cpu().numpy())
        records["all_agegroup"].append(agegroup.cpu().numpy())
        
        # 計算 loss 與預測
        if self.training_mode in ["ERM", "DS", "RW"]:
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights
            total_loss = per_sample_loss.mean()
            predicted_logits = logits_debiased

        elif self.training_mode == "GroupDRO":
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights

            gender_hard_labels = gender_hard_labels.to(device)
            unique_genders = torch.unique(gender_hard_labels)
            loss_g_list = []
            for g in unique_genders:
                mask = (gender_hard_labels == g)
                if mask.sum() > 0 and g.item() != -1:
                    group_loss = per_sample_loss[mask].mean()
                    loss_g_list.append((group_loss + self.lambda_GDRO / math.sqrt(self.gender_count[g.item()]), g.item()))
            if len(loss_g_list) == 0:
                total_loss = per_sample_loss.mean()
            else:
                worst_L_g, worst_g = max(loss_g_list, key=lambda x: x[0])
                total_loss = worst_L_g

            predicted_logits = logits_debiased
        
        elif self.training_mode == "GR":
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights
            classification_loss = per_sample_loss.mean()

            prediction_distribution = F.softmax(logits_debiased, dim=1)
            predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
            labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

            gender_hard_labels = gender_hard_labels.to(device)
            unique_genders = torch.unique(gender_hard_labels)

            TPR_diffs = []
            FPR_diffs = []
            F1_diffs = []
            eps = 1e-8
            B, C = labels_binary.shape
            for c in range(C):
                pred_c = predictions_binary[:, c]
                label_c = labels_binary[:, c]

                gender_metrics = {}
                for g in unique_genders:
                    g_int = int(g.item())
                    mask = (gender_hard_labels == g)
                    TP_g = (pred_c * label_c * mask).sum()
                    FN_g = ((1 - pred_c) * label_c * mask).sum()
                    FP_g = (pred_c * (1 - label_c) * mask).sum()
                    TN_g = ((1 - pred_c) * (1 - label_c) * mask).sum()

                    TPR_g = TP_g / (TP_g + FN_g + eps)
                    FPR_g = FP_g / (FP_g + TN_g + eps)

                    gender_metrics[g_int] = (TPR_g, FPR_g)
                keys = list(gender_metrics.keys())
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        TPR_diffs.append(abs(gender_metrics[keys[i]][0] - gender_metrics[keys[j]][0]))
                        FPR_diffs.append(abs(gender_metrics[keys[i]][1] - gender_metrics[keys[j]][1]))
                # 此處不計算 F1 gap，僅以 TPR/FPR gap 示意

            if len(TPR_diffs) == 0:
                TPR_RMS_gap = torch.tensor(0.0, device=device)
            else:
                TPR_RMS_gap = torch.sqrt(torch.mean(torch.tensor(TPR_diffs, device=device)**2))

            if len(FPR_diffs) == 0:
                FPR_RMS_gap = torch.tensor(0.0, device=device)
            else:
                FPR_RMS_gap = torch.sqrt(torch.mean(torch.tensor(FPR_diffs, device=device)**2))

            if self.GR_target == "TPR+FPR":
                GR_loss = self.lambda_GR * (TPR_RMS_gap + FPR_RMS_gap)
            elif self.GR_target == "TPR":
                GR_loss = self.lambda_GR * TPR_RMS_gap
            elif self.GR_target == "FPR":
                GR_loss = self.lambda_GR * FPR_RMS_gap
            else:
                raise NotImplementedError(f"Unknown GR target: {self.GR_target}")

            total_loss = classification_loss + GR_loss
            if "GR_loss" not in records:
                records["GR_loss"] = []
            if "emotion_loss" not in records:
                records["emotion_loss"] = []
            records["GR_loss"].append(GR_loss)
            records["emotion_loss"].append(classification_loss)
            predicted_logits = logits_debiased
        
        elif self.training_mode == "BPA":
            # Balanced Pseudo-cluster Adaption
            cluster_ids = gender_hard_labels  # 採用 pseudo group (K_Means)
            unique_clusters = torch.unique(cluster_ids)

            cluster_loss_map = {}
            for c in unique_clusters:
                mask = (cluster_ids == c)
                if mask.sum() == 0:
                    continue
                avg_loss_c = per_sample_loss[mask].mean()
                cluster_loss_map[c.item()] = avg_loss_c.detach()

            for c, avg_loss_c in cluster_loss_map.items():
                old_w = self.cluster_importance[c]
                new_w = self.bpa_ema_alpha * old_w + (1 - self.bpa_ema_alpha) * avg_loss_c.item()
                self.cluster_importance[c] = new_w

            sum_w = sum([self.cluster_importance[c] for c in unique_clusters])
            if sum_w < 1e-8:
                sample_weight_bpa = torch.ones_like(per_sample_loss, device=device)
            else:
                sample_weight_bpa = torch.zeros_like(per_sample_loss, device=device)
                for c in unique_clusters:
                    mask = (cluster_ids == c)
                    sample_weight_bpa[mask] = self.cluster_importance[c] / sum_w

            final_weight_each = sample_weight_bpa * sample_weights
            total_loss = (final_weight_each * per_sample_loss).sum()
            predicted_logits = logits_debiased
        else:
            raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

        # 計算並記錄預測結果
        prediction_distribution = F.softmax(predicted_logits, dim=1)
        predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
        labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

        if "all_predictions_binary" not in records:
            records["all_predictions_binary"] = []
        if "all_labels_binary" not in records:
            records["all_labels_binary"] = []
        if "all_genders" not in records:
            records["all_genders"] = []
        records["all_predictions_binary"].append(predictions_binary.cpu().numpy())
        records["all_labels_binary"].append(labels_binary.cpu().numpy())
        records["all_genders"].append(gender_hard_labels.cpu().numpy())

        records["loss"].append(total_loss.item())
        records["filename"] += filenames

        # 將預測結果轉為 emotion 字串以利後續記錄
        all_emotions_np = np.array(self.all_emotions)
        for idx in range(len(labels_binary)):
            true_emo = ";".join(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True)==1.0)[0]])
            pred_emo = ";".join(all_emotions_np[np.where(predictions_binary[idx].cpu().numpy(force=True)==1.0)[0]])
            if "truth" not in records:
                records["truth"] = []
            if "predict" not in records:
                records["predict"] = []
            records["truth"].append(true_emo)
            records["predict"].append(pred_emo)

        return total_loss

    def log_records(self, mode, records, logger, global_step, **kwargs):
        # 計算 macro-f1 與 accuracy
        all_preds = np.concatenate(records["all_predictions_binary"], axis=0)  # (N, C)
        all_labels = np.concatenate(records["all_labels_binary"], axis=0)      # (N, C)

        report_dict = classification_report(all_labels, all_preds, target_names=self.all_emotions, output_dict=True)
        macro_f1 = report_dict['macro avg']['f1-score']

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
            accuracy_c = (TP + TN) / denom if denom > 0 else 0.0
            acc_list.append(accuracy_c)
        acc = np.mean(acc_list) if len(acc_list) > 0 else 0.0

        average_loss = torch.FloatTensor(records['loss']).mean().item()

        metrics_to_log = {
            'macro-f1': macro_f1,
            'acc': acc,
            'loss': average_loss
        }
        if self.training_mode == "GR":
            GR_loss = torch.FloatTensor(records['GR_loss']).mean()
            metrics_to_log['GR_loss'] = GR_loss

        for key, val in metrics_to_log.items():
            logger.add_scalar(f'emotion-{self.fold}/{mode}-{key}', val, global_step=global_step)
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} {key}: {val}")
                f.write(f'{mode} {key} at step {global_step}: {val}\n')
            if key == 'loss' and mode == 'dev' and val < self.best_score and ((self.start_saving_ckpt_step is None) or self.start_saving_ckpt_step < global_step):
                self.best_score = torch.ones(1)*val
                with open(Path(self.expdir) / "log.log", 'a') as f:
                    f.write(f'New best on {mode} {key} at step {global_step}: {val}\n')
                best_ckpt = f'{mode}-best.ckpt'
                logger.add_text('best_ckpt', best_ckpt, global_step=global_step)

        # 計算 bias 指標：分別基於「真實性別」、「種族」與「年齡群」
        if mode in ["dev", "test"]:
            # 1. 基於真實性別
            all_real_gender = np.concatenate(records["all_real_gender"], axis=0)
            unique_real_gender = np.unique(all_real_gender)
            TPR_diffs_gender, FPR_diffs_gender, F1_diffs_gender = [], [], []
            for c in range(C):
                pred_c = all_preds[:, c]
                label_c = all_labels[:, c]
                metrics = {}
                for g in unique_real_gender:
                    mask = (all_real_gender == g)
                    pred_g = pred_c[mask]
                    label_g = label_c[mask]
                    TP = np.sum((pred_g == 1) & (label_g == 1))
                    FP = np.sum((pred_g == 1) & (label_g == 0))
                    FN = np.sum((pred_g == 0) & (label_g == 1))
                    TN = np.sum((pred_g == 0) & (label_g == 0))
                    TPR = safe_div(TP, TP+FN)
                    FPR = safe_div(FP, FP+TN)
                    precision = safe_div(TP, TP+FP)
                    recall = TPR
                    f1 = safe_div(2*precision*recall, precision+recall) if (precision+recall)>0 else 0.0
                    metrics[g] = (TPR, FPR, f1)
                keys = list(metrics.keys())
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        TPR_diffs_gender.append(abs(metrics[keys[i]][0] - metrics[keys[j]][0]))
                        FPR_diffs_gender.append(abs(metrics[keys[i]][1] - metrics[keys[j]][1]))
                        F1_diffs_gender.append(abs(metrics[keys[i]][2] - metrics[keys[j]][2]))
            rms_tpr_gap_gender = rms_gap(TPR_diffs_gender)
            rms_fpr_gap_gender = rms_gap(FPR_diffs_gender)
            rms_f1_gap_gender = rms_gap(F1_diffs_gender)
            DP_disparities_gender = []
            for c in range(C):
                pred_c = all_preds[:, c]
                global_pos = safe_div(np.sum(pred_c==1), len(pred_c))
                dp_vals = []
                for g in unique_real_gender:
                    mask = (all_real_gender == g)
                    pred_1_g = np.sum(pred_c[mask]==1)
                    n_g = np.sum(mask)
                    p_yhat = safe_div(pred_1_g, n_g)
                    dp_vals.append(abs(p_yhat - global_pos))
                DP_disparities_gender.append(np.max(dp_vals) if len(dp_vals)>0 else 0.0)
            rms_dp_gender = rms_gap(DP_disparities_gender)
            max_dp_gap_gender = np.max(DP_disparities_gender) if len(DP_disparities_gender)>0 else 0.0

            # 2. 基於種族（忽略 Unknown，即 race == -1）
            all_race = np.concatenate(records["all_race"], axis=0)
            valid_idx = (all_race != -1)
            if np.sum(valid_idx) == 0:
                rms_tpr_gap_race = 0.0
                rms_fpr_gap_race = 0.0
                rms_f1_gap_race = 0.0
                rms_dp_race = 0.0
            else:
                filtered_all_race = all_race[valid_idx]
                filtered_all_preds = all_preds[valid_idx, :]
                filtered_all_labels = all_labels[valid_idx, :]
                unique_race = np.unique(filtered_all_race)
                TPR_diffs_race, FPR_diffs_race, F1_diffs_race = [], [], []
                for c in range(C):
                    pred_c = filtered_all_preds[:, c]
                    label_c = filtered_all_labels[:, c]
                    metrics = {}
                    for r in unique_race:
                        mask = (filtered_all_race == r)
                        pred_r = pred_c[mask]
                        label_r = label_c[mask]
                        TP = np.sum((pred_r == 1) & (label_r == 1))
                        FP = np.sum((pred_r == 1) & (label_r == 0))
                        FN = np.sum((pred_r == 0) & (label_r == 1))
                        TN = np.sum((pred_r == 0) & (label_r == 0))
                        TPR = safe_div(TP, TP+FN)
                        FPR = safe_div(FP, FP+TN)
                        precision = safe_div(TP, TP+FP)
                        recall = TPR
                        f1 = safe_div(2*precision*recall, precision+recall) if (precision+recall)>0 else 0.0
                        metrics[r] = (TPR, FPR, f1)
                    keys = list(metrics.keys())
                    for i in range(len(keys)):
                        for j in range(i+1, len(keys)):
                            TPR_diffs_race.append(abs(metrics[keys[i]][0] - metrics[keys[j]][0]))
                            FPR_diffs_race.append(abs(metrics[keys[i]][1] - metrics[keys[j]][1]))
                            F1_diffs_race.append(abs(metrics[keys[i]][2] - metrics[keys[j]][2]))
                rms_tpr_gap_race = rms_gap(TPR_diffs_race)
                rms_fpr_gap_race = rms_gap(FPR_diffs_race)
                rms_f1_gap_race = rms_gap(F1_diffs_race)
                DP_disparities_race = []
                for c in range(C):
                    pred_c = filtered_all_preds[:, c]
                    global_pos = safe_div(np.sum(pred_c==1), len(pred_c))
                    dp_vals = []
                    for r in unique_race:
                        mask = (filtered_all_race == r)
                        pred_1_r = np.sum(pred_c[mask]==1)
                        n_r = np.sum(mask)
                        p_yhat = safe_div(pred_1_r, n_r)
                        dp_vals.append(abs(p_yhat - global_pos))
                    DP_disparities_race.append(np.max(dp_vals) if len(dp_vals)>0 else 0.0)
                rms_dp_race = rms_gap(DP_disparities_race)
                max_dp_gap_race = np.max(DP_disparities_race) if len(DP_disparities_race)>0 else 0.0

            # 3. 基於年齡群
            all_agegroup = np.concatenate(records["all_agegroup"], axis=0)
            unique_agegroup = np.unique(all_agegroup)
            TPR_diffs_agegroup, FPR_diffs_agegroup, F1_diffs_agegroup = [], [], []
            for c in range(C):
                pred_c = all_preds[:, c]
                label_c = all_labels[:, c]
                metrics = {}
                for a in unique_agegroup:
                    mask = (all_agegroup == a)
                    pred_a = pred_c[mask]
                    label_a = label_c[mask]
                    TP = np.sum((pred_a == 1) & (label_a == 1))
                    FP = np.sum((pred_a == 1) & (label_a == 0))
                    FN = np.sum((pred_a == 0) & (label_a == 1))
                    TN = np.sum((pred_a == 0) & (label_a == 0))
                    TPR = safe_div(TP, TP+FN)
                    FPR = safe_div(FP, FP+TN)
                    precision = safe_div(TP, TP+FP)
                    recall = TPR
                    f1 = safe_div(2*precision*recall, precision+recall) if (precision+recall)>0 else 0.0
                    metrics[a] = (TPR, FPR, f1)
                keys = list(metrics.keys())
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        TPR_diffs_agegroup.append(abs(metrics[keys[i]][0] - metrics[keys[j]][0]))
                        FPR_diffs_agegroup.append(abs(metrics[keys[i]][1] - metrics[keys[j]][1]))
                        F1_diffs_agegroup.append(abs(metrics[keys[i]][2] - metrics[keys[j]][2]))
            rms_tpr_gap_agegroup = rms_gap(TPR_diffs_agegroup)
            rms_fpr_gap_agegroup = rms_gap(FPR_diffs_agegroup)
            rms_f1_gap_agegroup = rms_gap(F1_diffs_agegroup)
            DP_disparities_agegroup = []
            for c in range(C):
                pred_c = all_preds[:, c]
                global_pos = safe_div(np.sum(pred_c==1), len(pred_c))
                dp_vals = []
                for a in unique_agegroup:
                    mask = (all_agegroup == a)
                    pred_1_a = np.sum(pred_c[mask]==1)
                    n_a = np.sum(mask)
                    p_yhat = safe_div(pred_1_a, n_a)
                    dp_vals.append(abs(p_yhat - global_pos))
                DP_disparities_agegroup.append(np.max(dp_vals) if len(dp_vals)>0 else 0.0)
            rms_dp_agegroup = rms_gap(DP_disparities_agegroup)
            max_dp_gap_agegroup = np.max(DP_disparities_agegroup) if len(DP_disparities_agegroup)>0 else 0.0

            # 將各項 bias 指標寫入 logger
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_tpr_gap_gender', rms_tpr_gap_gender, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_fpr_gap_gender', rms_fpr_gap_gender, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_f1_gap_gender', rms_f1_gap_gender, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_dp_gap_gender', rms_dp_gender, global_step=global_step)

            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_tpr_gap_race', rms_tpr_gap_race, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_fpr_gap_race', rms_fpr_gap_race, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_f1_gap_race', rms_f1_gap_race, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_dp_gap_race', rms_dp_race, global_step=global_step)

            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_tpr_gap_agegroup', rms_tpr_gap_agegroup, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_fpr_gap_agegroup', rms_fpr_gap_agegroup, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_f1_gap_agegroup', rms_f1_gap_agegroup, global_step=global_step)
            logger.add_scalar(f'emotion-{self.fold}/{mode}-rms_dp_gap_agegroup', rms_dp_agegroup, global_step=global_step)

            with open(Path(self.expdir) / "log.log", 'a') as f:
                f.write(f"{mode} bias metrics on real gender: rms_tpr_gap={rms_tpr_gap_gender}, rms_fpr_gap={rms_fpr_gap_gender}, rms_f1_gap={rms_f1_gap_gender}, rms_dp={rms_dp_gender}\n")
                f.write(f"{mode} bias metrics on race: rms_tpr_gap={rms_tpr_gap_race}, rms_fpr_gap={rms_fpr_gap_race}, rms_f1_gap={rms_f1_gap_race}, rms_dp={rms_dp_race}\n")
                f.write(f"{mode} bias metrics on agegroup: rms_tpr_gap={rms_tpr_gap_agegroup}, rms_fpr_gap={rms_fpr_gap_agegroup}, rms_f1_gap={rms_f1_gap_agegroup}, rms_dp={rms_dp_agegroup}\n")
        return []
