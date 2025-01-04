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
from sklearn.metrics import classification_report

from .dataset import prepare_datasets, collate_fn_padd
from ..model import *
from collections import defaultdict

# Suppress warnings for cleaner log outputs
warnings.filterwarnings("ignore")

# ======
# 損失函數
# ======

def class_balanced_softmax_cross_entropy_with_softtarget(logits, targets, weights, reduction='mean'):
    """
    使用 class-balanced 權重的 soft cross entropy 損失計算。

    Args:
        logits (Tensor): 預測結果 (batch_size, num_classes)
        targets (Tensor): 標籤的 soft one-hot 向量 (batch_size, num_classes)
        weights (Tensor): 每個類別的權重 (num_classes)
        reduction (str): 損失縮減方式，可為 'mean', 'sum', 'none'

    Returns:
        Tensor: 計算後的損失值
    """
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

# ======
# 梯度反轉層 (對抗式學習)
# ======

class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反轉 (Gradient Reversal Layer, GRL)

    用於對抗式訓練中，將梯度的方向反轉，讓特徵提取器不能輕易預測某些敏感屬性 (例如性別)。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    """
    梯度反轉層模組。
    """
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DownstreamExpert(nn.Module):
    """
    使用對抗式訓練的情緒識別模型。

    此模型同時學習情緒分類與對抗性地阻止模型從特徵中預測性別。
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        
        # 輸入維度與設定
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        # 建立資料路徑
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
        
        # 載入模型設定
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        # 前置投影層
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])

        # 主模型 (情緒預測)
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=len(self.config['categorical']["emo_type"]),
            **model_conf,
        )

        # 損失函數 (情緒)
        self.objective = class_balanced_softmax_cross_entropy_with_softtarget
        self.num_adversarial_layers = downstream_expert['debias'].get('num_adversarial_layers', 1)
        self.lambda_diff = downstream_expert['debias'].get('lambda_diff', 0.1)  # λ_diff hyperparameter
        self.lambda_adv = downstream_expert['debias'].get('lambda_adv', 0.8) 
        # 對抗式性別分類器 (使用2層線性層)
        self.grl = GradientReversal(alpha=1.0)
        self.adv_encoders = nn.ModuleList()
        for _ in range(self.num_adversarial_layers):
            encoder = nn.Sequential(
                nn.Linear(self.modelrc['projector_dim'], self.modelrc['projector_dim']),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.modelrc['projector_dim'], self.modelrc['projector_dim'] // 2),
            )
            self.adv_encoders.append(encoder)
        self.adv_classifiers = nn.ModuleList()
        for _ in range(self.num_adversarial_layers):
            classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.modelrc['projector_dim'] // 2, 1)
            )
            self.adv_classifiers.append(classifier)

        self.gender_criterion = nn.BCEWithLogitsLoss() 

        self.expdir = expdir
        self.register_buffer('best_score', torch.ones(1) * 99999)
        
        gender_count = defaultdict(int)
        for idx in range(len(self.train_dataset)):
            wav, lab, utt, g = self.train_dataset[idx]
            gender_count[g] += 1
        
        total_genders = len(gender_count)
        # Compute gender-specific weights
        self.gender_weights = {g: (len(self.train_dataset) / total_genders / count) 
                          for g, count in gender_count.items()}

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

    def _collate_wrapper(self, batch):
        total_wav, total_lab, total_utt, total_gender = collate_fn_padd(batch)
        return total_wav, total_lab, total_utt, total_gender

    def forward(self, mode, features, labels, filenames, gender_labels, records, **kwargs):
        """
        Args:
            mode (str): 訓練模式 (train/dev/test)
            features (list[Tensor]): 每個樣本的聲音特徵
            labels (Tensor): 真實情緒標籤 (soft target)
            filenames (list[str]): 該batch中樣本的檔名
            records (dict): 紀錄各種統計與結果的字典
            gender_labels (Tensor): 真實性別標籤 (若無則產生預設值)

        Returns:
            Tensor: 損失值
        """

        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device)

        # 將 variable-length features pad 並投影
        padded_features = pad_sequence(features, batch_first=True)
        projected_features = self.projector(padded_features)

        # 預測情緒
        predicted_logits, hidden_states = self.model(projected_features, features_len)
        labels = labels.to(device)
        emotion_loss = self.objective(predicted_logits, labels, self.class_balanced_weights.to(device), reduction='mean')

        gender_labels = gender_labels.to(device)
        
        # Compute adversarial losses for each layer
        # Only if valid labels are present
        total_adv_loss = 0.0
        collected_adv_features = []  # Will store h_A (adv_feature) for difference loss calculation
        correct_gender_preds = 0
        total_gender_samples = 0

        for idx in range(self.num_adversarial_layers):
            # Check if all are -1 (no valid labels)
            if (gender_labels == -1).all():
                # Skip if no valid labels
                continue
            else:
                # Use only valid entries
                valid_mask = (gender_labels != -1)
                valid_features = projected_features[valid_mask]
                valid_labels = gender_labels[valid_mask]

                # Adversarial prediction
                adv_features = torch.mean(valid_features, dim=1) 
                reversed_features = self.grl(adv_features)   
                adv_feature = self.adv_encoders[idx](reversed_features)
                adv_pred = self.adv_classifiers[idx](adv_feature)
                adv_loss = 0  # Initialize weighted adversarial loss
                for g, weight in self.gender_weights.items():
                    # Mask for current gender
                    gender_mask = (valid_labels == g)
                    if gender_mask.sum() > 0:
                        # Calculate loss for the current gender
                        gender_adv_loss = self.gender_criterion(
                            adv_pred[gender_mask].squeeze(1), 
                            valid_labels[gender_mask].float()
                        )
                        # Weight the loss
                        adv_loss += gender_adv_loss * weight
                total_adv_loss += adv_loss
                
                # Gender prediction accuracy
                gender_preds = (torch.sigmoid(adv_pred.squeeze(1)) > 0.5).float()
                correct_gender_preds += (gender_preds == valid_labels).sum().item()
                total_gender_samples += valid_labels.size(0)
                
                collected_adv_features.append(adv_feature)
        adverserial_loss = self.lambda_adv / self.num_adversarial_layers * total_adv_loss
        gender_accuracy = correct_gender_preds / total_gender_samples if total_gender_samples > 0 else 0.0
        difference_loss = 0.0
        if self.lambda_diff > 0.0 and len(collected_adv_features) > 1:
            # Suppose we have k adv_features: h_A_1, h_A_2, ..., h_A_k
            # Each h_A_i is (N_i, D) dimension. For difference loss, we consider pairs i != j.
            # We compute ||h_A_i^T h_A_j||_F^2
            # First, we might want to ensure the same batch size or handle if different sets have different sizes.
            # In this example, assume each adv_feature is computed on the same valid_mask, so N_i == N_j.
            # If they differ, you need additional logic.
            
            # We'll just sum over all pairs (i,j), i!=j
            for i in range(len(collected_adv_features)):
                for j in range(i+1, len(collected_adv_features)):
                    if i != j:
                        # h_A_i: (N, D)
                        # h_A_i^T h_A_j: (D, D)
                        # Frobenius norm squared: sum of squared elements
                        inter = collected_adv_features[i].t() @ collected_adv_features[j]
                        difference_loss += inter.pow(2).sum()

            difference_loss = self.lambda_diff * difference_loss

        # 合併損失
        total_loss = emotion_loss + adverserial_loss + difference_loss

        # 計算預測結果
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
            records["emotion_loss"] = []
            records["adv_loss"] = []
            records["diff_loss"] = []
            records["gender_accuracy"] = []

        records["all_predictions_binary"].append(predictions_binary.cpu().numpy())
        records["all_labels_binary"].append(labels_binary.cpu().numpy())
        if gender_labels is not None:
            records["all_genders"].append(gender_labels.cpu().numpy())
        else:
            records["all_genders"].append(np.zeros((len(labels_binary),), dtype=np.int64))

        # Store loss for later averaging
        records["loss"].append(total_loss.item())
        records["emotion_loss"].append(emotion_loss.item())
        records["adverserial_loss"].append(adverserial_loss.item())
        records["difference_loss"].append(difference_loss)
        records["filename"] += filenames
        records["gender_accuracy"].append(gender_accuracy)

        # 將預測結果及真實情緒寫入紀錄
        all_emotions_np = np.array(self.all_emotions)
        for idx in range(len(labels_binary)):
            true_emo = ";".join(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True) == 1.0)[0]])
            pred_emo = ";".join(all_emotions_np[np.where(predictions_binary[idx].cpu().numpy(force=True) == 1.0)[0]])
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
            'loss': average_loss,
            'emotion_loss': torch.FloatTensor(records['emotion_loss']).mean().item(),
            'adverserial_loss': torch.FloatTensor(records['adverserial_loss']).mean().item(),
            'difference_loss': torch.FloatTensor(records['difference_loss']).mean().item(),
            'gender_accuracy': torch.FloatTensor(records['gender_accuracy']).mean().item(),
        }

        save_names = []
        for key, val in metrics_to_log.items():
            logger.add_scalar(f'emotion-{self.fold}/{mode}-{key}', val, global_step=global_step)
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} {key}: {val}")
                f.write(f'{mode} {key} at step {global_step}: {val}\n')
            if key == 'emotion_loss' and mode == 'dev' and val < self.best_score:
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
                print(f"{mode} TPR RMS gap: {rms_tpr}, max gap: {max_tpr}")
                f.write(f"{mode} TPR RMS gap: {rms_tpr}, max gap: {max_tpr}\n")
                print(f"{mode} FPR RMS gap: {rms_fpr}, max gap: {max_fpr}")
                f.write(f"{mode} FPR RMS gap: {rms_fpr}, max gap: {max_fpr}\n")
                print(f"{mode} F1 RMS gap: {rms_f1}, max gap: {max_f1}")
                f.write(f"{mode} F1 RMS gap: {rms_f1}, max gap: {max_f1}\n")
                print(f"{mode} DP RMS disparity: {rms_dp}, max disparity: {max_dp}")
                f.write(f"{mode} DP RMS disparity: {rms_dp}, max disparity: {max_dp}\n")

            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fname} {pred}\n" for fname, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fname} {tr}\n" for fname, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names
