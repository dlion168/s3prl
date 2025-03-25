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
    - "LfF": Learning from Failure
    - "SiH": Signal is Harder than bias
    - "DisEnt": Disentangling-based Debiasing
    - "LVR": Latent Variance Regularization
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.training_mode = downstream_expert['debias'].get('training_mode', 'ERM')

        # LfF 相關超參數 (例如 q 值, EMA decay等)
        self.lff_q = downstream_expert['debias'].get('lff_q', 0.7)
        self.lff_ema_alpha = 0.7  # <- 固定 exponential decay 超參數
        
        self.sih_q = downstream_expert['debias'].get('sih_q', 0.7)
        
        self.lambda_dis_focal = downstream_expert['debias'].get('lambda_dis_focal', 5.0)
        self.lambda_swap = downstream_expert['debias'].get('lambda_swap', 1.0)
        self.disent_t_swap = downstream_expert['debias'].get('disent_t_swap', 10000)

        self.lambda_GR = downstream_expert['debias'].get('lambda_GR', 1.0)
        self.lambda_GDRO = downstream_expert['debias'].get('lambda_GDRO', 0.0)
        
        # -------------- LVR 相關超參數 --------------
        self.omega_LVR = downstream_expert['debias'].get('omega_LVR', 0.3)
        self.lambda_LVR = downstream_expert['debias'].get('lambda_LVR', 0.1)
        self.enable_center_cls = downstream_expert['debias'].get('enable_center_cls', True)
        self.lvr_previous_centers = None
        # ------------------------------------------
        self.sihlvr_t_start = downstream_expert['debias'].get('sihlvr_t_start', 0)
        
        # ---------- BLIND 相關超參數 -----------
        # gamma_blind: focal re-weighting exponent
        self.gamma_blind = downstream_expert['debias'].get('gamma_blind', 2.0)
        # 是否同時計算輔助分類器 (demographic or success) 的訓練損失
        self.lambda_aux = downstream_expert['debias'].get('lambda_aux', 1)

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
        
        # -------------【新增】讀取 config.json 的 GR_target 設定 -------------
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        self.GR_target = self.config.get('GR_target', 'TPR+FPR')  
        # 預設為 'TPR+FPR'，可在 config.json 裡設定為 "TPR" 或 "FPR"
        # -------------------------------------------------------------

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        # ---------------------- Debiased Model ----------------------
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = 2*self.modelrc['projector_dim'] if self.training_mode == "DisEnt" else self.modelrc['projector_dim'],
            output_dim = len(self.config['categorical']["emo_type"]),
            **model_conf,
        )

        # --------------------- Biased Model (for LfF) ---------------------
        self.projector_b = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model_b = model_cls(
            input_dim = 2*self.modelrc['projector_dim'] if self.training_mode == "DisEnt" else self.modelrc['projector_dim'],
            output_dim=len(self.config['categorical']["emo_type"]),
            **model_conf,
        )

        # LfF 執行時，我們會在 forward() 中用到的 EMA buffer (更新於每個 batch)
        self.ema_loss_b = []
        self.ema_loss_d = []

        # 給其他模式 (ERM / RW / DS / GroupDRO / GR) 用的損失函式
        self.objective = class_balanced_softmax_cross_entropy_with_softtarget
        
        self.blind_aux = None  # 預設 None
        if self.training_mode == "BLIND+d" or self.training_mode == "BLIND-d":
            # BLIND with demographic => 預設做 "binary classification" for gender
            # 也可視情況做多類別 => 這裡假設 gender_labels=0 or 1
            self.blind_aux = nn.Sequential(
                nn.Linear(self.modelrc['projector_dim'], 1),
            )

        self.expdir = expdir
        self.register_buffer('best_score', torch.ones(1)*99999)

        # Apply debiasing if chosen
        self.apply_debiasing(self.training_mode)
        self.start_saving_ckpt_step = kwargs['start_saving_ckpt_step']

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')

    def apply_debiasing(self, method):
        """
        For DS (downsampling), we subset the dataset.
        For RW (reweighting), we wrap the dataset with WeightedDataset using computed weights.
        For ERM/GroupDRO/GR/LfF: uniform weights=1
        """
        # Collect group info
        class_gender_pairs = []
        for idx in range(len(self.train_dataset)):
            wav, lab, utt, gender = self.train_dataset[idx]
            class_idx = np.argmax(lab)
            class_gender_pairs.append((class_idx, gender, idx))
        
        dev_class_gender_pairs = []
        for idx in range(len(self.dev_dataset)):
            wav, lab, utt, gender = self.dev_dataset[idx]
            class_idx = np.argmax(lab)
            dev_class_gender_pairs.append((class_idx, gender, idx))

        counts = defaultdict(int)
        class_counts = defaultdict(int)
        class_genders = defaultdict(lambda: defaultdict(list))
        dev_class_genders = defaultdict(lambda: defaultdict(list))
        self.gender_count = defaultdict(int)
        
        for (c, g, i) in class_gender_pairs:
            counts[(c,g)] += 1
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
                        group_count = counts[(c,g)]
                        w = (sum_c / G_c) * (1.0 / group_count)
                        for idx_sample in idx_list:
                            weight_dict[idx_sample] = w
                        for dev_idx_sample in dev_class_genders[c][g]:
                            dev_weight_dict[dev_idx_sample] = w

                self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
                self.dev_dataset = WeightedDataset(self.dev_dataset, dev_weight_dict)
            print("[RW] Assigned reweighting to training samples.")

        else:
            # ERM, GroupDRO, GR, LfF: just assign uniform weights=1
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

        # Debiased Model forward
        if self.training_mode != "DisEnt":
            logits_debiased, _ = self.model(projected_features, features_len)
        
        if self.training_mode in ["LfF", "SiH", "DisEnt", "SiHLVR"]:
            batch_id = kwargs['batch_id']
            biased_features_len = torch.IntTensor([len(feat) for feat in kwargs['addi_features']]).to(device)
            biased_padded_features = pad_sequence(kwargs['addi_features'], batch_first=True).to(device)
            projected_b = self.projector_b(biased_padded_features)
            if self.training_mode != "DisEnt":
                # 1) Biased Model forward
                logits_b, _ = self.model_b(projected_b, biased_features_len)
        
        # ------------------------- LfF 模式 --------------------------
        if self.training_mode == "LfF":
            # GCE loss (class-balanced)
            gce_loss_b = class_balanced_generalized_cross_entropy_loss(
                logits_b, 
                labels, 
                self.class_balanced_weights.to(device), 
                q=self.lff_q, 
                reduction='none'
            )  # (B,)
            loss_biased = gce_loss_b.mean()

            # 2) Debiased Model loss (Weighted CE)
            ce_loss_b = class_balanced_softmax_cross_entropy_with_softtarget(
                logits_b,
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )  # (B,)
            ce_loss_d = class_balanced_softmax_cross_entropy_with_softtarget(
                logits_debiased,
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )  # (B,)

            # ====== EMA Trick for stable training ======
            while len(self.ema_loss_b) <= batch_id:
                self.ema_loss_b.append(None)
                self.ema_loss_d.append(None)
                
            weights = torch.ones(ce_loss_d.shape[0]).to(device)
            if mode == 'train':
                # 若本 batch_i 尚未初始化，就直接設成本 batch 的 mean loss
                if self.ema_loss_b[batch_id] is None:
                    self.ema_loss_b[batch_id] = ce_loss_b.detach().clone()
                    self.ema_loss_d[batch_id] = ce_loss_d.detach().clone()
                else:
                    # Exponential Moving Average
                    alpha = self.lff_ema_alpha
                    old_b = self.ema_loss_b[batch_id]
                    old_d = self.ema_loss_d[batch_id]
                    self.ema_loss_b[batch_id] = alpha * old_b + (1 - alpha) * ce_loss_b.detach()  # (B, )
                    self.ema_loss_d[batch_id] = alpha * old_d + (1 - alpha) * ce_loss_d.detach()  # (B, )

                # 之後計算相對難度
                denom = self.ema_loss_b[batch_id] + self.ema_loss_d[batch_id] + 1e-8
                weights = self.ema_loss_b[batch_id] / denom
                weights = weights.to(device)
                # =====================================================
            
            # Weighted Cross Entropy for Debiased Model
            loss_debiased = (weights * ce_loss_d).mean()

            records["LfF_debiased_loss"].append(loss_debiased)
            records["LfF_biased_loss"].append(loss_biased)
            
            total_loss = loss_biased + loss_debiased
            predicted_logits = logits_debiased
            
        # ------------------------- 新增: SiH 模式 --------------------------
        elif self.training_mode == "SiH":
            """
            1) 偏置模型用 GCE 訓練 (與 LfF 類似)
            2) 對無偏模型的 cross entropy，再乘上一個 focal-like reweight (1 - p_b)^q
               其中 p_b 為偏置模型對正確類別（labels=1）所輸出的概率總和。
               而 (1 - p_b) 會先 detach()，以免影響偏置模型參數的更新。
            """
            gce_loss_b = class_balanced_generalized_cross_entropy_loss(
                logits_b, 
                labels, 
                self.class_balanced_weights.to(device), 
                q=self.lff_q, 
                reduction='none'
            )
            loss_biased = gce_loss_b.mean()

            # 2) Debiased model with focal-like weighting
            ce_loss_d = class_balanced_softmax_cross_entropy_with_softtarget(
                logits_debiased,
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )

            with torch.no_grad():
                prob_b = F.softmax(logits_b, dim=1)     # (B, C)
                # 取出正確類別的機率 (multi-label 時可將所有正確類別的 prob 累加)
                correct_prob_b = torch.sum(prob_b * labels, dim=1)  # (B,)
                # focal-like weighting factor
                weight_factor = (1.0 - correct_prob_b).detach().pow(self.sih_q)

            loss_debiased = (weight_factor * ce_loss_d).mean()

            records["SiH_debiased_loss"].append(loss_debiased.item())
            records["SiH_biased_loss"].append(loss_biased.item())

            total_loss = loss_biased + loss_debiased
            predicted_logits = logits_debiased
                # >>> NEW: SiHLVR <<<
        elif self.training_mode == "SiHLVR":
            """
            合併 SiH 與 LVR:
              - 偏置模型 (logits_b) 用 GCE 訓練 -> loss_biased
              - 去偏模型 (logits_debiased) 用 focal-like reweight = (1 - p_b)^q 乘上 cross entropy
              - 再在去偏模型的 latent space 做 LVR => total_loss = loss_biased + reweighted_cls + lambda_LVR * L_r + L_c
            """
            # 1) 偏置模型 GCE
            gce_loss_b = class_balanced_generalized_cross_entropy_loss(
                logits_b, 
                labels, 
                self.class_balanced_weights.to(device), 
                q=self.lff_q, 
                reduction='none'
            )
            loss_biased = gce_loss_b.mean()

            if kwargs['global_step'] < self.sihlvr_t_start:
                # >>> NEW for SiHLVR condition <<<
                # 只訓練 biased model
                total_loss = loss_biased
                # 不做 focal reweight nor LVR
                records["SiHLVR_biased_loss"].append(loss_biased.item())
                records["SiHLVR_debiased_loss"].append(10.0)
                records["SiHLVR_lvr_loss"].append(0.0)
                if self.enable_center_cls:
                    records["SiHLVR_center_loss"].append(0.0)

                # 預設用偏置模型 logits_b 當作「最終輸出」(若您想仍用 debiased 的 logits 也可)
                predicted_logits = logits_b

            else:
                # focal-like weighting on debiased model
                ce_loss_d = class_balanced_softmax_cross_entropy_with_softtarget(
                    logits_debiased,
                    labels,
                    self.class_balanced_weights.to(device),
                    reduction='none'
                )
                with torch.no_grad():
                    prob_b = F.softmax(logits_b, dim=1)
                    correct_prob_b = torch.sum(prob_b * labels, dim=1)
                    weight_factor = (1.0 - correct_prob_b).detach().pow(self.sih_q)

                classification_loss = (weight_factor * ce_loss_d).mean()

                # ---------- LVR part (對 debiased model) ----------
                B, C = labels.shape
                avg_features = projected_features.mean(dim=1)
                if self.lvr_previous_centers is None:
                    centers = labels.T @ avg_features
                    self.lvr_previous_centers = centers.detach().cpu()
                else:
                    prev_center = self.lvr_previous_centers.to(device)
                    centers = labels.T @ avg_features
                    centers = (1.0 - self.omega_LVR) * centers + self.omega_LVR * prev_center

                diff = avg_features.unsqueeze(1) - centers.unsqueeze(0)
                dist_sq = diff.pow(2).mean(dim=2)
                L_r = (dist_sq * labels).sum()

                L_c = torch.tensor(0.0, device=device)
                if self.enable_center_cls:
                    center_features = []
                    center_labels = []
                    for c_idx in range(C):
                        center_c = centers[c_idx]
                        center_features.append(center_c)
                        oh = torch.zeros(C, device=device)
                        oh[c_idx] = 1.0
                        center_labels.append(oh)

                    center_features = torch.stack(center_features, dim=0).unsqueeze(1)
                    center_len = torch.ones(center_features.size(0), dtype=torch.int32, device=device)
                    logits_center, _ = self.model(center_features, center_len)
                    center_labels = torch.stack(center_labels, dim=0)
                    L_c = self.objective(
                        logits_center,
                        center_labels,
                        self.class_balanced_weights.to(device),
                        reduction='mean'
                    )

                self.lvr_previous_centers = centers.detach().cpu()

                L_lvr = L_r + L_c
                lvr_part = self.lambda_LVR * L_lvr
                total_loss = loss_biased + classification_loss + lvr_part

                records["SiHLVR_biased_loss"].append(loss_biased.item())
                records["SiHLVR_debiased_loss"].append(classification_loss.item())
                records["SiHLVR_lvr_loss"].append(L_r.item())
                if self.enable_center_cls:
                    records["SiHLVR_center_loss"].append(L_c.item())

                predicted_logits = logits_debiased
        
        elif self.training_mode == "DisEnt":
            # 使 Ci 主要基於 zi，故 zb 在 concat 時做 detach，不回傳梯度給 Eb
            # 同理，Cb 主要基於 zb，故 zi 在 concat 時做 detach。
            zi_for_Ci = torch.cat([projected_features, projected_b.detach()], dim=-1)  # (B, T, 2D)
            zb_for_Cb = torch.cat([projected_features.detach(), projected_b], dim=-1)  # (B, T, 2D)

            # 接著分別餵給 model_i, model_b (對應 Ci, Cb) 做分類輸出
            logits_debiased, _ = self.model(zi_for_Ci, features_len)
            logits_b, _ = self.model_b(zb_for_Cb, features_len)
            # === 2) 計算 relative difficulty score W(z) (Eq. (1)) ===
            # CE(Ci(z), y) & CE(Cb(z), y)
            ce_ci = class_balanced_softmax_cross_entropy_with_softtarget(
                logits_debiased,  # Ci(z)
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )  # shape=(B,)

            ce_cb = class_balanced_softmax_cross_entropy_with_softtarget(
                logits_b,         # Cb(z)
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )  # shape=(B,)
            
                        # ====== EMA Trick for stable training ======
            while len(self.ema_loss_b) <= batch_id:
                self.ema_loss_b.append(None)
                self.ema_loss_d.append(None)

            weights = torch.ones(ce_cb.shape[0]).to(device)
            
            if mode == 'train':
                # 若本 batch_i 尚未初始化，就直接設成本 batch 的 mean loss
                if self.ema_loss_b[batch_id] is None:
                    self.ema_loss_b[batch_id] = ce_cb.detach().clone()
                    self.ema_loss_d[batch_id] = ce_ci.detach().clone()
                else:
                    # Exponential Moving Average
                    alpha = self.lff_ema_alpha
                    old_b = self.ema_loss_b[batch_id]
                    old_d = self.ema_loss_d[batch_id]
                    self.ema_loss_b[batch_id] = alpha * old_b + (1 - alpha) * ce_cb.detach() # shape=(B,)
                    self.ema_loss_d[batch_id] = alpha * old_d + (1 - alpha) * ce_ci.detach() # shape=(B,)

                # 之後計算相對難度
                denom = self.ema_loss_b[batch_id] + self.ema_loss_d[batch_id] + 1e-8         # shape=(B,)
                weights = self.ema_loss_b[batch_id] / denom                                  # shape=(B,)
                weights = weights.to(device)

            # === 3) L_dis = W(z)*CE(Ci(z),y) + lambda_dis * GCE(Cb(z),y) ===
            ce_ci_weighted = weights * ce_ci  # shape=(B,)
            gce_cb = class_balanced_generalized_cross_entropy_loss(
                logits_b,
                labels,
                self.class_balanced_weights.to(device),
                q=0.7,  # 也可從 config 取
                reduction='none'
            )  # shape=(B,)

            L_dis = ce_ci_weighted.mean() + self.lambda_dis_focal * gce_cb.mean()
            
            # === 4) 若 iteration > t_swap，才執行 swap => zswap = [zi; z̃b] 並計算 L_swap ===
            #     先確定 batch size >= 2，才能亂序 permute；若 batch=1 就不做 swap 了
            L_swap = torch.tensor(0.0, device=device)       
            ce_ci_swap_weighted = torch.tensor(0.0, device=device) 
            gce_cb_swap = torch.tensor(0.0, device=device)
            if (kwargs['global_step'] > self.disent_t_swap) and (len(features) > 1):
                B = projected_features.size(0)
                # 隨機打亂 zb，perm_idx 不含自己 => (簡化起見，這裡直接用 random.shuffle)
                perm_idx = torch.randperm(B, device=device)
                # 取得 z̃b
                zb_perm = projected_b[perm_idx, :]  # (B, T, D)
                
                # zswap 只需要丟給 Ci, Cb 分別做 forward:
                #   Ci(zswap) => Ci([zi, z̃b])  => 這裡簡化為 concat 再 forward
                #   Cb(zswap) => Cb([zi, z̃b])，
                # 這裡展示最簡方式：直接 cat 在 feature dim。
                zswap_for_Ci = torch.cat([projected_features, zb_perm.detach()], dim=-1)  # shape=(B, T, D*2)
                zswap_for_Cb = torch.cat([projected_features.detach(), zb_perm], dim=-1)  # shape=(B, T, D*2)

                # forward Ci(zswap)
                logits_ci_swap, _ = self.model(zswap_for_Ci, features_len)
                # forward Cb(zswap)
                logits_cb_swap, _ = self.model_b(zswap_for_Cb, features_len)

                # W(z) (同一個 W_z) 也可或不可重算，論文中是對每個樣本 z 都有 W(z)，
                # 但這裡示意就用同一個 W_z. 

                # ỹ: 若有「針對 permute 後 bias 属性」的 label，可從 kwargs['perm_labels'] 取
                # 沒有的話，先假設一樣都是 labels
                perm_labels = labels[perm_idx, :]

                ce_ci_swap = class_balanced_softmax_cross_entropy_with_softtarget(
                    logits_ci_swap,
                    labels,
                    self.class_balanced_weights.to(device),
                    reduction='none'
                )  # shape=(B,)

                gce_cb_swap = class_balanced_generalized_cross_entropy_loss(
                    logits_cb_swap,
                    perm_labels,  # 論文中對 Cb 用 ỹ
                    self.class_balanced_weights.to(device),
                    q=0.7,
                    reduction='none'
                )  # shape=(B,)

                # L_swap = W(z)*CE(Ci(zswap), y) + lambda_swap_b * GCE(Cb(zswap), ỹ)
                # 這裡假設 lambda_swap_b = self.lambda_swap (也可做細分)
                ce_ci_swap_weighted = weights * ce_ci_swap
                L_swap = ce_ci_swap_weighted.mean() + self.lambda_dis_focal * gce_cb_swap.mean()

            total_loss = L_dis + self.lambda_swap * L_swap  # Eq. (4)
            predicted_logits = logits_debiased  # 最終輸出用 Ci(z) (intrinsic) 的結果

            # 為了觀察訓練情況，也可記錄:
            records["DisEnt_Ldis_unbiased"].append(ce_ci_weighted.mean().item())
            records["DisEnt_Ldis_biased"].append(gce_cb.mean().item())
            records["DisEnt_Ldis"].append(L_dis.item())
            records["DisEnt_Lswap_unbiased"].append(ce_ci_swap_weighted.mean().item())
            records["DisEnt_Lswap_biased"].append(gce_cb_swap.mean().item())
            records["DisEnt_Lswap"].append(L_swap.item())
        
        # --------------------- BLIND + demographic (BLIND+d) ---------------------
        elif self.training_mode == "BLIND+d":
            """
            1) 使用一個輔助分類器來預測 demographic (gender_label)，
               gender_labels: 0 或 1 (假設二元).
            2) 根據該分類器的預測信心水平 down-weight "易於推斷 demographic" 的樣本.
            3) 使用 Debiased Focal Loss (DFL) 形式:
                 L = (1 - confidence_of_demo)^gamma * CE(main_model)
               並且對輔助分類器本身也做一個 cross entropy，最後合併:
                 total_loss = main_loss + lambda_aux * aux_loss
            """
            # --- 先做主模型 logits ---
            per_sample_ce = self.objective(
                logits_debiased,
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )  # shape=(B,)

            # --- 建構並 forward 輔助分類器 (demo) ---
            # 這裡只示意用 [B, D] = mean over time steps
            # 或自行改成 pooled feature
            if self.blind_aux is None:
                raise RuntimeError("BLIND+d mode needs self.blind_aux defined.")
            rep_for_demo = projected_features.mean(dim=1)  # shape=(B, D)
            logits_demo = self.blind_aux(rep_for_demo).squeeze(-1)  # shape=(B,)

            # demo_label: 0/1 => shape=(B,)
            # 在整份程式中 gender_labels 已傳入. 若 -1 代表未知, 這裡假設 batch 中都有有效 gender
            # 只示意 => 要排除 -1 的樣本可能得特別處理
            demo_label = gender_labels.to(device).float()

            # (a) aux_loss：預測性別 => binary cross-entropy
            aux_prob = torch.sigmoid(logits_demo)
            # Identify valid demographic samples (demo_label != -1)
            valid_mask = (demo_label != -1.0)

            # If we have valid samples, apply the BLIND weighting only to them
            if valid_mask.sum() > 0:
                # Confidence for valid portion
                confidence_demo_valid = torch.where(
                    demo_label[valid_mask] > 0.5,
                    aux_prob[valid_mask],
                    1.0 - aux_prob[valid_mask]
                )
                weighting_factor_valid = (1.0 - confidence_demo_valid).detach().pow(self.gamma_blind)
                
                # Weighted main loss on valid samples
                main_loss_valid = weighting_factor_valid * per_sample_ce[valid_mask]

                # For invalid samples (demo_label == -1), do normal CE
                main_loss_invalid = per_sample_ce[~valid_mask]

                # Combine them across the entire batch
                B = per_sample_ce.shape[0]
                main_loss = (main_loss_valid.sum() + main_loss_invalid.sum()) / float(B)

                # Auxiliary loss only on valid samples
                aux_loss_valid = F.binary_cross_entropy(
                    aux_prob[valid_mask],
                    demo_label[valid_mask],
                    reduction='none'
                ).mean()

                total_loss = main_loss + self.lambda_aux * aux_loss_valid

                records["BLINDd_aux_loss"].append(aux_loss_valid.item())
                records["BLINDd_main_loss"].append(main_loss.item())

            else:
                # If no valid sample in this batch, fallback to normal CE
                main_loss = per_sample_ce.mean()
                total_loss = main_loss

            predicted_logits = logits_debiased
        
        # --------------------- BLIND - demographic (BLIND-d) ---------------------
        elif self.training_mode == "BLIND-d":
            """
            1) 輔助分類器不再預測 demographic，而是預測「該樣本是否會被主模型正確分類」(success=1 / fail=0).
            2) 同樣以 (1 - success_prob)^gamma 來對主模型的 CE 進行 down-weight.
            3) total_loss = main_loss + lambda_aux * success_loss
            """
            if self.blind_aux is None:
                raise RuntimeError("BLIND-d mode needs self.blind_aux defined.")
            
            # (A) 先算主模型的 CE
            per_sample_ce = self.objective(
                logits_debiased,
                labels,
                self.class_balanced_weights.to(device),
                reduction='none'
            )  # shape=(B,)

            # (B) 計算每個樣本的 per-sample accuracy
            #   - pred_bin, lbl_bin 都是 (B, C)
            pred_dist = F.softmax(logits_debiased, dim=1)
            pred_bin = torch.where(pred_dist > self.k_thresold, 1.0, 0.0)
            lbl_bin = torch.where(labels > self.k_thresold, 1.0, 0.0)

            B, C = pred_bin.size()
            matching = (pred_bin == lbl_bin).sum(dim=1)    # shape=(B,)
            acc_i = matching.float() / float(C)            # shape=(B,)

            # (C) success detector forward => 預測每筆樣本的 accuracy
            rep_for_success = projected_features.mean(dim=1)  # shape=(B, D)
            logits_success = self.blind_aux(rep_for_success).squeeze(-1)  # shape=(B,)
            success_prob = torch.sigmoid(logits_success)  # ∈ (0,1)

            # (C1) success_loss => 用 MSE 擬合 f1_i
            success_loss = F.binary_cross_entropy(
                success_prob, acc_i.to(device), reduction='mean'
            )
            
            # (D) 依 success_prob 產生 re-weight: w_i = (1 - success_prob_i)^gamma
            weighting_factor = (1.0 - success_prob).detach().pow(self.gamma_blind)

            main_loss = (weighting_factor * per_sample_ce).mean()
            total_loss = main_loss + self.lambda_aux * success_loss

            # 紀錄
            records["BLINDd_success_loss"].append(success_loss.item())
            records["BLINDd_main_loss"].append(main_loss.item())

            predicted_logits = logits_debiased


        # --------------------- 其他模式: ERM / DS / RW ---------------------
        elif self.training_mode in ["ERM", "DS", "RW"]:
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights
            total_loss = per_sample_loss.mean()
            predicted_logits = logits_debiased

        # -------------------- 其他模式: GroupDRO ---------------------------
        elif self.training_mode == "GroupDRO":
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights

            gender_labels = gender_labels.to(device)
            unique_genders = torch.unique(gender_labels)
            loss_g_list = []
            for g in unique_genders:
                mask = (gender_labels == g)
                if mask.sum() > 0 and g.item() != -1:
                    group_loss = per_sample_loss[mask].mean()
                    loss_g_list.append((group_loss + self.lambda_GDRO / math.sqrt(self.gender_count[g.item()]), g.item()))
            if len(loss_g_list) == 0 and g.item() != -1:
                total_loss = per_sample_loss.mean()
            else:
                worst_L_g, worst_g = max(loss_g_list, key=lambda x: x[0])
                total_loss = worst_L_g

            predicted_logits = logits_debiased

        # -------------------- 其他模式: GR ---------------------------
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

            gender_labels = gender_labels.to(device)
            unique_genders = torch.unique(gender_labels)

            TPR_diffs = []
            FPR_diffs = []
            eps = 1e-8
            B, C = labels_binary.shape
            for c in range(C):
                pred_c = predictions_binary[:, c]
                label_c = labels_binary[:, c]

                gender_TPR = {}
                gender_FPR = {}

                for g in unique_genders:
                    g_int = int(g.item())
                    mask = (gender_labels == g).float()
                    TP_g = (pred_c * label_c * mask).sum()
                    FN_g = ((1 - pred_c) * label_c * mask).sum()
                    FP_g = (pred_c * (1 - label_c) * mask).sum()
                    TN_g = ((1 - pred_c) * (1 - label_c) * mask).sum()

                    TPR_g = TP_g / (TP_g + FN_g + eps)
                    FPR_g = FP_g / (FP_g + TN_g + eps)

                    gender_TPR[g_int] = TPR_g
                    gender_FPR[g_int] = FPR_g

                g_list = list(gender_TPR.keys())
                for i in range(len(g_list)):
                    for j in range(i+1, len(g_list)):
                        g1, g2 = g_list[i], g_list[j]
                        TPR_diff = (gender_TPR[g1] - gender_TPR[g2]).abs()
                        FPR_diff = (gender_FPR[g1] - gender_FPR[g2]).abs()
                        TPR_diffs.append(TPR_diff)
                        FPR_diffs.append(FPR_diff)

            if len(TPR_diffs) == 0:
                TPR_RMS_gap = torch.tensor(0.0, device=device)
            else:
                TPR_diffs_tensor = torch.stack(TPR_diffs)
                TPR_RMS_gap = torch.sqrt((TPR_diffs_tensor**2).mean())

            if len(FPR_diffs) == 0:
                FPR_RMS_gap = torch.tensor(0.0, device=device)
            else:
                FPR_diffs_tensor = torch.stack(FPR_diffs)
                FPR_RMS_gap = torch.sqrt((FPR_diffs_tensor**2).mean())

            if self.GR_target == "TPR+FPR":
                GR_loss = self.lambda_GR * (TPR_RMS_gap + FPR_RMS_gap)
            elif self.GR_target == "TPR":
                GR_loss = self.lambda_GR * TPR_RMS_gap
            elif self.GR_target == "FPR":
                GR_loss = self.lambda_GR * FPR_RMS_gap
            else:
                raise NotImplementedError(f"Unknown GR target: {self.GR_target}")

            total_loss = classification_loss + GR_loss
            records["GR_loss"].append(GR_loss)
            records["emotion_loss"].append(classification_loss)
            predicted_logits = logits_debiased
            
        elif self.training_mode == "LVR":
            """
            1) 先做 classification loss
            2) 為每個類別 c 計算 batch center (avgZ_c)，然後與前一 batch 的 center 做線性插值: 
               C_i^b = (1 - ω)*avgZ_c + ω*C_i^{b-1}
            3) 計算 L_r = ∑_i ∑_r ∑_j ( z_jr^i - c_j^i )^2, 
               (這裡對所有屬於類別 i 的樣本 r 做 L2-distance 到 center i)
            4) (可選) L_c: 把各 center 再丟進 model 做分類
            5) total_loss = classification_loss + lambda_LVR * L_r + L_c
            """
            # (1) Classification loss
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights
            classification_loss = per_sample_loss.mean()

            B, C = labels.shape

            # (2) 計算 batch center + 平滑
            avg_features = projected_features.mean(dim=1) #(B, H)
            avgZ = labels.T @ avg_features
            prev_center = self.lvr_previous_centers.to(device) if self.lvr_previous_centers != None else avgZ
            centers = (1.0 - self.omega_LVR)*avgZ + self.omega_LVR*prev_center

            # (3) 計算 regularization loss L_r
            # (3-1) 在 batch 維度 (B) 和 label/class 維度 (C) 進行廣播
            #     diff 形狀會是 [B, C, H]
            diff = avg_features.unsqueeze(1) - centers.unsqueeze(0)
            # (3-2) 對 H 維度做平均 => dist_sq 形狀 [B, C]
            dist_sq = diff.pow(2).mean(dim=2)
            # (3-3) 乘上 labels (形狀 [B, C])，再進行整體加總
            L_r = (dist_sq * labels).sum()

            # (4) (可選) center 的分類 L_c
            L_c = torch.tensor(0.0, device=device)
            if self.enable_center_cls:
                center_features = []
                center_labels = []
                for c_idx in range(C):
                    center_c = centers[c_idx]
                    center_features.append(center_c)
                    oh = torch.zeros(C, device=device)
                    oh[c_idx] = 1.0
                    center_labels.append(oh)

                if len(center_features) > 0:
                    center_features = torch.stack(center_features, dim=0).unsqueeze(1)  # (num_centers, 1, H)
                    center_len = torch.ones(center_features.size(0), dtype=torch.int32, device=device)
                    logits_center, _ = self.model(center_features, center_len)
                    center_labels = torch.stack(center_labels, dim=0)      # (num_centers, C)
                    L_c = self.objective(
                        logits_center,
                        center_labels,
                        self.class_balanced_weights.to(device),
                        reduction='mean'
                    )

            # (5) 總損失
            total_loss = classification_loss + self.lambda_LVR * L_r + L_c
            predicted_logits = logits_debiased

            # (6) 更新 self.lvr_previous_centers
            # 若該類別存在於本 batch，則存入 centers[c_idx]
            # 存 CPU 以免顯存累積
            self.lvr_previous_centers = centers.detach().cpu()

            # 額外記錄以便觀察
            records["LVR_loss"] = records.get("LVR_loss", [])
            records["LVR_loss"].append(L_r.item())
            if self.enable_center_cls:
                records["LVR_center_loss"] = records.get("LVR_center_loss", [])
                records["LVR_center_loss"].append(L_c.item())
        elif self.training_mode == "random":
            per_sample_loss = self.objective(
                logits_debiased, 
                labels, 
                self.class_balanced_weights.to(device), 
                reduction='none'
            )
            per_sample_loss = per_sample_loss * sample_weights
            total_loss = per_sample_loss.mean()
            predicted_logits = torch.rand(logits_debiased.shape).to(device)
        else:
            raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

        # 統一計算並記錄預測結果
        prediction_distribution = F.softmax(predicted_logits, dim=1)
        predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
        labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

        records["all_predictions_binary"].append(predictions_binary.cpu().numpy())
        records["all_labels_binary"].append(labels_binary.cpu().numpy())
        if gender_labels is not None:
            records["all_genders"].append(gender_labels.cpu().numpy())
        else:
            records["all_genders"].append(np.zeros((len(labels_binary),), dtype=np.int64))

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
        all_labels = np.concatenate(records["all_labels_binary"], axis=0)      # (N, C)

        reprot_dict = classification_report(all_labels, all_preds, target_names=self.all_emotions, output_dict=True)
        macro_f1 = reprot_dict['macro avg']['f1-score']

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
        elif self.training_mode == "LVR":
            LVR_loss = torch.FloatTensor(records['LVR_loss']).mean()
            LVR_center_loss = torch.FloatTensor(records['LVR_center_loss']).mean()
            metrics_to_log['LVR_loss'] = LVR_loss
            metrics_to_log['LVR_center_loss'] = LVR_center_loss
        elif self.training_mode == "LfF":
            LfF_debiased_loss = torch.FloatTensor(records['LfF_debiased_loss']).mean()
            LfF_biased_loss = torch.FloatTensor(records['LfF_biased_loss']).mean()
            metrics_to_log['LfF_debiased_loss'] = LfF_debiased_loss
            metrics_to_log['LfF_biased_loss'] = LfF_biased_loss
        elif self.training_mode == "SiH":
            SiH_debiased_loss = torch.FloatTensor(records['SiH_debiased_loss']).mean()
            SiH_biased_loss = torch.FloatTensor(records['SiH_biased_loss']).mean()
            metrics_to_log['SiH_debiased_loss'] = SiH_debiased_loss
            metrics_to_log['SiH_biased_loss'] = SiH_biased_loss
        elif self.training_mode == "SiHLVR":
            # 這裡我們會在 forward() 裡記錄 4 個list: ["SiHLVR_biased_loss", "SiHLVR_debiased_loss", "SiHLVR_lvr_loss", "SiHLVR_center_loss"]
            SiHLVR_biased_loss = torch.FloatTensor(records['SiHLVR_biased_loss']).mean()
            SiHLVR_debiased_loss = torch.FloatTensor(records['SiHLVR_debiased_loss']).mean()
            SiHLVR_lvr_loss = torch.FloatTensor(records['SiHLVR_lvr_loss']).mean()
            # center_loss 只有在 enable_center_cls == True 時才會有
            SiHLVR_center_loss = 0.0
            if "SiHLVR_center_loss" in records:
                SiHLVR_center_loss = torch.FloatTensor(records['SiHLVR_center_loss']).mean()

            metrics_to_log['SiHLVR_biased_loss'] = SiHLVR_biased_loss
            metrics_to_log['SiHLVR_debiased_loss'] = SiHLVR_debiased_loss
            metrics_to_log['SiHLVR_LVR_loss'] = SiHLVR_lvr_loss
            metrics_to_log['SiHLVR_center_loss'] = SiHLVR_center_loss
        elif self.training_mode == "DisEnt":
            DisEnt_Ldis_unbiased = torch.FloatTensor(records["DisEnt_Ldis_unbiased"]).mean()
            DisEnt_Ldis_biased = torch.FloatTensor(records["DisEnt_Ldis_biased"]).mean()
            DisEnt_Lswap_unbiased = torch.FloatTensor(records["DisEnt_Lswap_unbiased"]).mean()
            DisEnt_Lswap_biased = torch.FloatTensor(records["DisEnt_Lswap_biased"]).mean()
            DisEnt_Ldis = torch.FloatTensor(records['DisEnt_Ldis']).mean()
            DisEnt_Lswap = torch.FloatTensor(records['DisEnt_Lswap']).mean()
            metrics_to_log['DisEnt_Ldis'] = DisEnt_Ldis
            metrics_to_log['DisEnt_Lswap'] = DisEnt_Lswap
            metrics_to_log["DisEnt_Ldis_unbiased"] = DisEnt_Ldis_unbiased
            metrics_to_log["DisEnt_Ldis_biased"] = DisEnt_Ldis_biased
            metrics_to_log["DisEnt_Lswap_unbiased"] = DisEnt_Lswap_unbiased
            metrics_to_log["DisEnt_Lswap_biased"] = DisEnt_Lswap_biased
        elif self.training_mode == "BLIND+d":
            # BLIND+d
            if "BLINDd_aux_loss" in records:
                aux_loss_val = torch.FloatTensor(records["BLINDd_aux_loss"]).mean()
                main_loss_val = torch.FloatTensor(records["BLINDd_main_loss"]).mean()
                metrics_to_log["BLINDd_aux_loss"] = aux_loss_val
                metrics_to_log["BLINDd_main_loss"] = main_loss_val
        elif self.training_mode == "BLIND-d":
            # BLIND-d
            if "BLINDd_success_loss" in records:
                success_loss_val = torch.FloatTensor(records["BLINDd_success_loss"]).mean()
                main_loss_val = torch.FloatTensor(records["BLINDd_main_loss"]).mean()
                metrics_to_log["BLINDd_success_loss"] = success_loss_val
                metrics_to_log["BLINDd_main_loss"] = main_loss_val

        save_names = []
        for key, val in metrics_to_log.items():
            logger.add_scalar(f'emotion-{self.fold}/{mode}-{key}', val, global_step=global_step)
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} {key}: {val}")
                f.write(f'{mode} {key} at step {global_step}: {val}\n')
            if key == 'loss' and mode == 'dev' and val < self.best_score and ((self.start_saving_ckpt_step is None) or self.start_saving_ckpt_step < global_step):
                self.best_score = torch.ones(1)*val
                with open(Path(self.expdir) / "log.log", 'a') as f:
                    f.write(f'New best on {mode} {key} at step {global_step}: {val}\n')
                save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            all_genders = np.concatenate(records["all_genders"], axis=0)  # (N,)
            unique_genders = np.unique(all_genders)

            TPR_diffs = []
            FPR_diffs = []
            F1_diffs = []

            for c in range(C):
                pred_c = all_preds[:, c]
                label_c = all_labels[:, c]

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
