# -*- coding: utf-8 -*-
"""
    FileName     [ dataset.py ]
    Synopsis     [ The emotion classifier dataset, now also returns gender labels and always returns sample_weights. ]
"""

import json
import glob
import os
import csv
import pickle as pk
import numpy as np
import torch
import torch.utils.data as torch_utils
import librosa
from tqdm import tqdm
from multiprocessing import Pool

SAMPLE_RATE = 16000

def get_norm_stat_for_wav(wav_list, verbose=False):
    count = 0
    wav_sum = 0
    wav_sqsum = 0
    iterator = tqdm(wav_list) if verbose else wav_list
    for cur_wav in iterator:
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav**2)
        count += len(cur_wav)
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)
    return wav_mean, wav_std

def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    return raw_wav

class WavExtractor:
    def __init__(self, wav_paths, nj=24):
        self.wav_path_list = wav_paths
        self.nj = nj

    def extract(self):
        print("Extracting wav files...")
        with Pool(self.nj) as p:
            wav_list = list(tqdm(p.imap(extract_wav, self.wav_path_list), total=len(self.wav_path_list)))
        return wav_list

class WavSet(torch_utils.Dataset):
    """
    WavSet now also stores gender labels.
    Returns (wav, lab, utt, gender) per item.
    Does not include sample_weights by itself; that will be handled by WeightedDataset.
    """
    def __init__(self, wav_list, lab_list, utt_list, gender_list,
                 print_dur=False, lab_type='categorical', 
                 wav_mean=None, wav_std=None, label_config=None):
        super(WavSet, self).__init__()
        self.wav_list = wav_list
        self.lab_list = lab_list
        self.utt_list = utt_list
        self.gender_list = gender_list
        self.print_dur = print_dur
        self.lab_type = lab_type
        self.label_config = label_config

        if self.lab_type == "categorical":
            assert len(self.label_config.get("emo_type", [])) != 0, "No 'emo_type' found in label config."

        self.max_dur = 10 * SAMPLE_RATE
        assert wav_mean is not None and wav_std is not None, "Please provide wav_mean and wav_std."
        self.wav_mean = wav_mean
        self.wav_std = wav_std

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = extract_wav(self.wav_list[idx])[:self.max_dur]
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std + 1e-6)
        cur_utt = self.utt_list[idx]
        cur_lab = self.lab_list[idx]
        cur_gender = self.gender_list[idx]  # 0: Male, 1: Female

        if self.print_dur:
            return cur_wav, cur_lab, cur_utt, len(cur_wav), cur_gender
        else:
            return cur_wav, cur_lab, cur_utt, cur_gender

def collate_fn_padd(batch):
    """
    Now we assume WeightedDataset is always used,
    so each item has (wav, lab, utt, gender, weight).
    """
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

def collate_fn(samples):
    return zip(*samples)

class DataManager:
    def __init__(self, env_path):
        self.env_dict = self.__load_env__(env_path)
        self.msp_label_dict = None
        self.msp_gender_dict = None

    def __load_env__(self, env_path):
        with open(env_path, 'r') as f:
            env_dict = json.load(f)
        return env_dict

    def get_wav_path(self, split_type=None, wav_loc=None, label_path=None):
        if split_type is None:
            wav_list = glob.glob(os.path.join(wav_loc, "*.wav"))
        else:
            utt_list = self.get_utt_list(split_type, label_path)
            wav_list = [os.path.join(wav_loc, utt_id) for utt_id in utt_list]
        wav_list.sort()
        return wav_list

    def get_utt_list(self, split_type, label_path):
        utt_list = []
        sid = self.env_dict["data_split_type"][split_type]
        with open(label_path, 'r') as f:
            f.readline()  # skip header
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                stype = row[-1]
                if stype == sid:
                    utt_list.append(utt_id)
        utt_list.sort()
        return utt_list

    def __load_msp_cat_label_dict__(self, label_path):
        self.msp_label_dict = dict()
        self.msp_gender_dict = dict()
        emo_class_list = self.get_categorical_emo_class()

        with open(label_path, 'r') as f:
            header = f.readline().strip().split(",")
            emo_idx_list = [header.index(emo) for emo in emo_class_list]
            gender_idx = header.index("Gender")
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                cur_emo_lab = [float(row[emo_idx]) for emo_idx in emo_idx_list]
                self.msp_label_dict[utt_id] = cur_emo_lab
                gender_str = row[gender_idx]
                if gender_str.lower() == "female":
                    gender_label = 1
                elif gender_str.lower() == "male":
                    gender_label = 0
                else:
                    gender_label = -1
                self.msp_gender_dict[utt_id] = gender_label

    def get_msp_labels_and_gender(self, utt_list, lab_type, label_path):
        if lab_type == "categorical":
            if self.msp_label_dict is None or self.msp_gender_dict is None:
                self.__load_msp_cat_label_dict__(label_path)
            emo_labels = np.array([self.msp_label_dict[utt_id] for utt_id in utt_list])
            gender_labels = np.array([self.msp_gender_dict[utt_id] for utt_id in utt_list])
            return emo_labels, gender_labels
        else:
            raise NotImplementedError("Only 'categorical' label type is implemented.")

    def get_categorical_emo_class(self):
        return self.env_dict["categorical"]["emo_type"]

    def get_label_config(self, label_type):
        assert label_type in ["categorical", "dimensional"]
        return self.env_dict[label_type]

def prepare_datasets(datarc, config_path):
    dam = DataManager(config_path)

    audio_path = os.path.join(datarc['root'], datarc['corpus'], "Audios")
    label_path = os.path.join(datarc['root'], datarc['corpus'], datarc['p_or_s'], 
                               "labels_consensus_" + datarc['test_fold'].replace("fold","") + ".csv")

    train_utts = dam.get_utt_list("train", label_path=label_path)
    dev_utts = dam.get_utt_list("dev", label_path=label_path)
    test_utts = dam.get_utt_list("test", label_path=label_path)

    train_wav_path = dam.get_wav_path("train", wav_loc=audio_path, label_path=label_path)
    dev_wav_path = dam.get_wav_path("dev", wav_loc=audio_path, label_path=label_path)
    test_wav_path = dam.get_wav_path("test", wav_loc=audio_path, label_path=label_path)

    train_labs, train_genders = dam.get_msp_labels_and_gender(train_utts, lab_type='categorical', label_path=label_path)
    dev_labs, dev_genders = dam.get_msp_labels_and_gender(dev_utts, lab_type='categorical', label_path=label_path)
    test_labs, test_genders = dam.get_msp_labels_and_gender(test_utts, lab_type='categorical', label_path=label_path)

    k_threshold = 1 / train_labs.shape[1]
    train_labs_tensor = torch.Tensor(train_labs)
    train_labs_binary = torch.where(train_labs_tensor > k_threshold, 1.0, 0.0)
    samples_per_cls = torch.sum(train_labs_binary, dim=0)

    beta = (train_labs.shape[0]-1)/train_labs.shape[0]
    no_of_classes = train_labs.shape[1]
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    class_balanced_weights = (weights / torch.sum(weights)) * no_of_classes

    train_wavs_np_path = os.path.join(datarc['root'], datarc['corpus'], datarc['p_or_s'], 
                                      "Train_wavs_numpy_" + datarc['test_fold'] + ".pkl")
    if not os.path.exists(train_wavs_np_path):
        print("Saving Wavs Numpy files:", train_wavs_np_path)
        train_wavs = WavExtractor(train_wav_path).extract()
        wav_mean, wav_std = get_norm_stat_for_wav(train_wavs)
        stats = {"wav_mean": wav_mean, "wav_std": wav_std}
        with open(train_wavs_np_path, 'wb') as f:
            pk.dump(stats, f)
    else:
        with open(train_wavs_np_path, 'rb') as f:
            stats = pk.load(f)
        wav_mean = stats["wav_mean"]
        wav_std = stats["wav_std"]

    label_config = dam.get_label_config(label_type='categorical')

    train_dataset = WavSet(train_wav_path, train_labs, train_utts, train_genders,
                           print_dur=False, lab_type='categorical',
                           label_config=label_config,
                           wav_mean=wav_mean, wav_std=wav_std)

    dev_dataset = WavSet(dev_wav_path, dev_labs, dev_utts, dev_genders,
                         print_dur=False, lab_type='categorical',
                         label_config=label_config,
                         wav_mean=wav_mean, wav_std=wav_std)

    test_dataset = WavSet(test_wav_path, test_labs, test_utts, test_genders,
                          print_dur=False, lab_type='categorical',
                          label_config=label_config,
                          wav_mean=wav_mean, wav_std=wav_std)

    categorical_emo = dam.get_categorical_emo_class()
    return train_dataset, dev_dataset, test_dataset, class_balanced_weights, k_threshold, categorical_emo

class WeightedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that returns sample weights along with original data.
    Here we assume base_dataset returns (wav, lab, utt, gender).
    This wrapper adds a weight for each sample.
    """

    def __init__(self, base_dataset, weight_dict):
        self.base_dataset = base_dataset
        self.weight_dict = weight_dict

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        wav, lab, utt, gender = self.base_dataset[idx]
        w = self.weight_dict[idx]
        return (wav, lab, utt, gender, w)
