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
         all_emotions) = prepare_datasets(self.datarc, self.config_path)
        
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
        self.num_adversarial_layers = self.modelrc.get('num_adversarial_layers', 1)
        self.lambda_diff = self.modelrc.get('lambda_diff', 0.0)  # λ_diff hyperparameter
        self.lambda_adv = self.modelrc.get('lambda_adv', 0.1) 
        # 對抗式性別分類器 (使用2層線性層)
        self.grl = GradientReversal(alpha=1.0)
        self.adv_encoders = nn.ModuleList()
        for _ in range(self.num_adversarial_layers):
            encoder = nn.Sequential(
                nn.Linear(self.modelrc['projector_dim'], self.modelrc['projector_dim'] // 2)
            )
            self.adv_encoders.append(encoder)
        self.adv_classifiers = nn.ModuleList()
        for _ in range(self.num_adversarial_layers):
            classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.modelrc['projector_dim'] // 2, 2)
            )
            self.adv_classifiers.append(classifier)

        self.gender_criterion = nn.CrossEntropyLoss()

        self.expdir = expdir
        self.register_buffer('best_score', torch.ones(1) * 99999)


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

    def forward(self, mode, features, labels, filenames, records, gender_labels=None, **kwargs):
        """
        前向傳播：
        1. 使用模型預測情緒分佈
        2. 使用梯度反轉層與性別分類器預測性別
        3. 損失 = 情緒損失 + 對抗損失 * lambda

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
                adv_loss = self.gender_criterion(adv_pred, valid_labels)
                total_adv_loss += adv_loss
                
                collected_adv_features.append(adv_feature)
        
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
        total_loss = emotion_loss + self.adversarial_lambda / self.num_adversarial_layers * total_adv_loss + difference_loss

        # 計算預測結果
        prediction_distribution = F.softmax(predicted_logits, dim=1)
        preditions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
        labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

        all_emotions = self.config['categorical']["emo_type"]
        report_dict = classification_report(labels_binary.cpu().numpy(force=True),
                                            preditions_binary.cpu().numpy(force=True),
                                            target_names=all_emotions,
                                            output_dict=True)
        records['acc'].append(report_dict['macro avg']['f1-score'])
        records['loss'].append(total_loss.item())
        records["filename"] += filenames

        # 將預測結果及真實情緒寫入紀錄
        all_emotions_np = np.array(all_emotions)
        predict_list = []
        truth_list = []
        for idx in range(len(labels_binary)):
            true_emo = ";".join(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True) == 1.0)[0]])
            pred_emo = ";".join(all_emotions_np[np.where(preditions_binary[idx].cpu().numpy(force=True) == 1.0)[0]])
            truth_list.append(true_emo)
            predict_list.append(pred_emo)

        records["predict"] += predict_list
        records["truth"] += truth_list

        return total_loss

    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(f'emotion-{self.fold}/{mode}-{key}', average, global_step=global_step)

            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'loss':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')
                    if mode == 'dev' and average < self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} {key} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
                elif key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fname} {pred}\n" for fname, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fname} {tr}\n" for fname, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names
