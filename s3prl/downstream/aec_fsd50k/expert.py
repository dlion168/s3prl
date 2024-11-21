import torch
#-------------#
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import FSD50KDataset, FSD50KFeatureDataset
from sklearn import metrics
from pathlib import Path
import numpy as np
import os 


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.pre_extract_dir = kwargs["pre_extract_dir"]
        
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"
        fold_mapping = {"fold1": 0, "fold2": 1, "fold3": 2, "fold4": 3, "fold5": 4}
        if self.fold not in fold_mapping:
            raise ValueError("fold_name must be one of 'fold1', 'fold2', 'fold3', 'fold4', 'fold5'")
        fold_idx = fold_mapping[self.fold]
        
        root_dir = self.datarc['file_path']
        
        # Identify folds
        folds = sorted([os.path.splitext(f)[0] for f in os.listdir(root_dir) if f.startswith("fold") and f.endswith(".json")])
        num_folds = len(folds)
        
        test_fold = folds[fold_idx]
        valid_fold = folds[(fold_idx + 1) % num_folds]
        train_folds = [f for f in folds if f not in (test_fold, valid_fold)]
        
        # Create datasets for each split
        self.train_dataset = FSD50KFeatureDataset(root_dir, self.pre_extract_dir, train_folds, sample_rate = kwargs["sample_rate"]
            ) if self.pre_extract_dir else FSD50KDataset(root_dir, train_folds, sample_rate = kwargs["sample_rate"])
        self.valid_dataset = FSD50KFeatureDataset(root_dir, self.pre_extract_dir, [valid_fold], sample_rate = kwargs["sample_rate"]
            ) if self.pre_extract_dir else FSD50KDataset(root_dir, [valid_fold], sample_rate = kwargs["sample_rate"])
        self.test_dataset = FSD50KFeatureDataset(root_dir, self.pre_extract_dir, [test_fold], sample_rate = kwargs["sample_rate"]
            ) if self.pre_extract_dir else FSD50KDataset(root_dir, [test_fold], sample_rate = kwargs["sample_rate"])
        
        self.expdir = expdir
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = len(self.train_dataset.class2id.keys()),
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))
        

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.valid_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)
        labels = torch.stack(labels).to(features.device)
        loss = self.objective(predicted, labels)
        # Deal with AUC-ROC and PR-ROC
        records['logit'] += predicted.cpu().float().tolist()
        records['labels'] += labels.cpu().float().tolist()
        records['loss'].append(loss.item())
        records['filename'] += filenames

        return loss
    
    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'aec_FSD50K/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_class"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_class"])]
                file.writelines(lines)

        return save_names
    
    def calculate_stats(output, target, class_indices=None):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      class_indices: list
        explicit indices of classes to calculate statistics for

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats