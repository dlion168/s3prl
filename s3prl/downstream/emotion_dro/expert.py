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

from .dataset import prepare_datasets, collate_fn_padd

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

class WeightedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that returns sample weights along with original data.
    """

    def __init__(self, base_dataset, weight_dict):
        self.base_dataset = base_dataset
        self.weight_dict = weight_dict

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        weight = self.weight_dict[idx]
        return (*item, weight)

def collate_fn_padd(batch):
    """
    Modified collate_fn to handle optional weight.
    If weight is present, batch items will have 5 elements.
    """
    has_weight = (len(batch[0]) == 5)
    if has_weight:
        total_wav = []
        total_lab = []
        total_utt = []
        total_gender = []
        total_weight = []
        for wav, lab, utt, gender, w in batch:
            total_wav.append(torch.Tensor(wav))
            total_lab.append(lab)
            total_utt.append(utt)
            total_gender.append(gender)
            total_weight.append(w)
        total_lab = torch.Tensor(np.asarray(total_lab))
        total_gender = torch.Tensor(total_gender).long()
        total_weight = torch.Tensor(total_weight)
        return total_wav, total_lab, total_utt, total_gender, total_weight
    else:
        total_wav = []
        total_lab = []
        total_utt = []
        total_gender = []
        for wav, lab, utt, gender in batch:
            total_wav.append(torch.Tensor(wav))
            total_lab.append(lab)
            total_utt.append(utt)
            total_gender.append(gender)
        total_lab = torch.Tensor(np.asarray(total_lab))
        total_gender = torch.Tensor(total_gender).long()
        return total_wav, total_lab, total_utt, total_gender

class DownstreamExpert(nn.Module):
    """
    training_mode (merged):
    - "ERM": Just ERM
    - "GroupDRO": Group Distributionally Robust Optimization
    - "DS": Downsampling debiasing
    - "RW": Reweighting debiasing

    We remove macro-f1 and acc recording in forward.
    We'll compute macro-f1 and acc in log_records.
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.training_mode = self.modelrc.get('training_mode', 'ERM')  
        # Possible values: "ERM", "GroupDRO", "DS", "RW"

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

        # If DS or RW is chosen, apply debiasing on the training dataset
        if self.training_mode in ["DS", "RW"]:
            self.apply_debiasing(self.training_mode)

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')

    def apply_debiasing(self, method):
        class_gender_pairs = []
        for idx in range(len(self.train_dataset)):
            wav, lab, utt, gender = self.train_dataset[idx]
            class_idx = np.argmax(lab)
            class_gender_pairs.append((class_idx, gender, idx))

        from collections import defaultdict
        counts = defaultdict(int)
        for (c, g, i) in class_gender_pairs:
            counts[(c,g)] += 1

        if method == "DS":
            # Downsampling
            if len(counts) == 0:
                return
            min_count = min(counts.values())
            group_samples = defaultdict(list)
            for (c,g,i) in class_gender_pairs:
                group_samples[(c,g)].append(i)
            new_indices = []
            for (c,g), idx_list in group_samples.items():
                random.shuffle(idx_list)
                chosen = idx_list[:min_count]
                new_indices.extend(chosen)
            self.train_dataset = Subset(self.train_dataset, new_indices)
            print(f"[DS] Downsampled training set to {len(new_indices)} samples.")

        elif method == "RW":
            # Reweighting
            if len(counts) == 0:
                return
            weights_map = {k:1.0/v for k,v in counts.items()}
            weight_dict = {}
            for (c,g,i) in class_gender_pairs:
                weight_dict[i] = weights_map[(c,g)]
            from . import WeightedDataset  # If needed, or define WeightedDataset above
            self.train_dataset = WeightedDataset(self.train_dataset, weight_dict)
            print("[RW] Assigned reweighting to training samples.")

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

    def forward(self, mode, features, labels, filenames, records, gender_labels=None, sample_weights=None, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device)
        padded_features = pad_sequence(features, batch_first=True).to(device)
        projected_features = self.projector(padded_features)
        predicted_logits, hidden_states = self.model(projected_features, features_len)
        labels = labels.to(device)

        per_sample_loss = self.objective(predicted_logits, labels, self.class_balanced_weights.to(device), reduction='none')

        # If RW is used, apply weights
        if self.training_mode == "RW" and sample_weights is not None:
            sample_weights = sample_weights.to(device)
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
            accuracy_c = (TP + TN) / (TP + TN + FP + FN)
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
            all_genders = np.concatenate(records["all_genders"], axis=0)            # (N,)

            def safe_div(a, b):
                return a / b if b > 0 else 0.0

            TPR_list, FPR_list, F1_list = [], [], []
            DP_disparities = []

            for c in range(C):
                pred_c = all_preds[:, c]
                label_c = all_labels[:, c]

                TP = np.sum((pred_c == 1) & (label_c == 1))
                FP = np.sum((pred_c == 1) & (label_c == 0))
                FN = np.sum((pred_c == 0) & (label_c == 1))
                TN = np.sum((pred_c == 0) & (label_c == 0))

                TPR = safe_div(TP, TP+FN)
                FPR = safe_div(FP, FP+TN)
                precision = safe_div(TP, TP+FP)
                recall = TPR
                f1 = safe_div(2*precision*recall, precision+recall) if (precision+recall)>0 else 0.0

                TPR_list.append(TPR)
                FPR_list.append(FPR)
                F1_list.append(f1)

                pred_1 = np.sum(pred_c==1)
                n_c = len(pred_c)
                global_pos = safe_div(pred_1, n_c)
                dp_vals = []
                unique_genders = np.unique(all_genders)
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

            def rms_gap(values):
                values = np.array(values)
                if len(values) == 0:
                    return 0.0
                mean_val = values.mean()
                diff = values - mean_val
                return math.sqrt(np.mean(diff**2))

            rms_tpr = rms_gap(TPR_list)
            rms_fpr = rms_gap(FPR_list)
            rms_f1 = rms_gap(F1_list)
            rms_dp = rms_gap(DP_disparities)

            max_tpr = max(TPR_list) if TPR_list else 0.0
            max_fpr = max(FPR_list) if FPR_list else 0.0
            max_f1 = max(F1_list) if F1_list else 0.0
            max_dp = max(DP_disparities) if DP_disparities else 0.0

            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} TPR RMS disparity: {rms_tpr}, max disparity: {max_tpr}")
                f.write(f"{mode} TPR RMS disparity: {rms_tpr}, max disparity: {max_tpr}\n")
                print(f"{mode} FPR RMS disparity: {rms_fpr}, max disparity: {max_fpr}")
                f.write(f"{mode} FPR RMS disparity: {rms_fpr}, max disparity: {max_fpr}\n")
                print(f"{mode} F1 RMS disparity: {rms_f1}, max disparity: {max_f1}")
                f.write(f"{mode} F1 RMS disparity: {rms_f1}, max disparity: {max_f1}\n")
                print(f"{mode} DP RMS disparity: {rms_dp}, max disparity: {max_dp}")
                f.write(f"{mode} DP RMS disparity: {rms_dp}, max disparity: {max_dp}\n")

            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fname} {pred}\n" for fname, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fname} {tr}\n" for fname, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names
