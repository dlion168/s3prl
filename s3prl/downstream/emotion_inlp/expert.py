import os
import math
import torch
import random
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

import json
import numpy as np
import warnings
import pickle as pk
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report

from .dataset import prepare_datasets, collate_fn_padd

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Dict
import scipy
from typing import List, Tuple
from tqdm import tqdm
import random
import warnings

# 匯入您提供的 get_debiasing_projection 和其他相關函式與類別
from src.classifier import SKlearnClassifier
from sklearn.linear_model import SGDClassifier

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)
    P_W = w_basis.dot(w_basis.T)
    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)
    return P

def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, by_class=False, Y_train_main=None,
                             Y_dev_main=None, dropout_rate = 0) -> Tuple[np.ndarray,List[np.ndarray],List[np.ndarray]]:
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

    I = np.eye(input_dim)

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers))
    for i in pbar:
        clf = SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1./(1 - dropout_rate + 1e-6)
        dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale

        relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
        relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: 
            continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:
            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    return P, rowspace_projections, Ws

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

class DownstreamExpert(nn.Module):
    """
    Implements Iterative Nullspace Projection (INLP) debias method.
    Also computes performance metrics (macro-f1, acc) and fairness metrics at log_records.
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.inlp_rounds = self.modelrc.get('inlp_rounds', 10)

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

        print("[INLP] Computing nullspace projection matrix P.")
        self.P = self.compute_inlp_projection()

    def _collate_wrapper(self, batch):
        return collate_fn_padd(batch)

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=self._collate_wrapper
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc['eval_batch_size'],
            shuffle=False,
            num_workers=self.datarc['num_workers'],
            collate_fn=self._collate_wrapper
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode):
        return getattr(self, f'get_{mode}_dataloader')()

    def compute_inlp_projection(self):
        # 收集訓練集特徵與敏感標籤
        train_loader = self.get_train_dataloader()
        X_list = []
        Z_list = []
        device = self.projector.weight.device
        with torch.no_grad():
            for wav, lab, utt, gender in train_loader:
                wav = [w.to(device) for w in wav]
                features_len = torch.IntTensor([len(f) for f in wav]).to(device)
                padded = pad_sequence(wav, batch_first=True)
                proj = self.projector(padded)
                proj_mean = proj.mean(dim=1).cpu().numpy()
                X_list.append(proj_mean)
                Z_list.append(gender.numpy())

        X = np.concatenate(X_list, axis=0)
        Z = np.concatenate(Z_list, axis=0)
        _, D = X.shape

        # 使用您提供的 get_debiasing_projection 函式來執行INLP
        # 這裡將 X, Z 當作訓練和開發資料
        # 在真實場景中，請準備獨立開發資料進行驗證
        P, rowspace_projections, Ws = get_debiasing_projection(
            classifier_class=SGDClassifier,
            cls_params={},
            num_classifiers=self.inlp_rounds,
            input_dim=D,
            is_autoregressive=True,
            min_accuracy=0.0,
            X_train=X,
            Y_train=Z,
            X_dev=X,
            Y_dev=Z,
            by_class=False
        )

        return torch.from_numpy(P).float().to(device)

    def forward(self, mode, features, labels, filenames, records, gender_labels=None, **kwargs):
        device = self.projector.weight.device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device)
        padded_features = pad_sequence(features, batch_first=True).to(device)
        projected_features = self.projector(padded_features)

        B, T, D = projected_features.shape
        reshaped = projected_features.reshape(B*T, D)
        reshaped = reshaped @ self.P.T
        projected_features = reshaped.reshape(B, T, D)

        predicted_logits, hidden_states = self.model(projected_features, features_len)
        labels = labels.to(device)

        per_sample_loss = self.objective(predicted_logits, labels, self.class_balanced_weights.to(device), reduction='none')
        total_loss = per_sample_loss.mean()

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

        records["all_predictions_binary"].append(predictions_binary.cpu().numpy())
        records["all_labels_binary"].append(labels_binary.cpu().numpy())
        if gender_labels is not None:
            records["all_genders"].append(gender_labels.cpu().numpy())
        else:
            records["all_genders"].append(np.zeros((len(labels_binary),), dtype=np.int64))

        records["filename"] += filenames

        all_emotions_np = np.array(self.all_emotions)
        for idx in range(len(labels_binary)):
            true_emo = ";".join(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True)==1.0)[0]])
            pred_emo = ";".join(all_emotions_np[np.where(predictions_binary[idx].cpu().numpy(force=True)==1.0)[0]])
            records["truth"].append(true_emo)
            records["predict"].append(pred_emo)

        return total_loss

    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        all_preds = np.concatenate(records["all_predictions_binary"], axis=0)
        all_labels = np.concatenate(records["all_labels_binary"], axis=0)
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
            accuracy_c = (TP + TN) / denom if denom>0 else 0
            acc_list.append(accuracy_c)
        acc = np.mean(acc_list) if len(acc_list) > 0 else 0.0

        if 'loss' in records:
            average_loss = torch.FloatTensor(records['loss']).mean().item()
        else:
            average_loss = 0.0

        metrics_to_log = {
            'macro-f1': macro_f1,
            'acc': acc,
            'loss': average_loss
        }

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
            all_genders = np.concatenate(records["all_genders"], axis=0)

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

            rms_tpr_gap = rms_gap(TPR_list)
            rms_fpr_gap = rms_gap(FPR_list)
            rms_f1_gap = rms_gap(F1_list)
            rms_dp_gap = rms_gap(DP_disparities)

            max_tpr_gap = np.max(TPR_list) - np.min(TPR_list) if len(TPR_list)>1 else 0.0
            max_fpr_gap = np.max(FPR_list) - np.min(FPR_list) if len(FPR_list)>1 else 0.0
            max_f1_gap = np.max(F1_list) - np.min(F1_list) if len(F1_list)>1 else 0.0
            max_dp_gap = np.max(DP_disparities) if len(DP_disparities)>0 else 0.0

            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} TPR RMS gap: {rms_tpr_gap}, max gap: {max_tpr_gap}")
                f.write(f"{mode} TPR RMS gap: {rms_tpr_gap}, max gap: {max_tpr_gap}\n")
                print(f"{mode} FPR RMS gap: {rms_fpr_gap}, max gap: {max_fpr_gap}")
                f.write(f"{mode} FPR RMS gap: {rms_fpr_gap}, max gap: {max_fpr_gap}\n")
                print(f"{mode} F1 RMS gap: {rms_f1_gap}, max gap: {max_f1_gap}")
                f.write(f"{mode} F1 RMS gap: {rms_f1_gap}, max gap: {max_f1_gap}\n")
                print(f"{mode} DP RMS disparity: {rms_dp_gap}, max disparity: {max_dp_gap}")
                f.write(f"{mode} DP RMS disparity: {rms_dp_gap}, max disparity: {max_dp_gap}\n")

            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fname} {pred}\n" for fname, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fname} {tr}\n" for fname, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names
