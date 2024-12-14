import numpy as np
import random
import itertools

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch


class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)
    
    
class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values in FairBatch.

        
    """
    def __init__(self, model, x_tensor, y_tensor, z_tensor, batch_size, alpha, target_fairness, replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.model = model
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z_data = z_tensor
        
        self.alpha = alpha
        self.fairness_type = target_fairness
        self.replacement = replacement
        
        self.N = len(z_tensor)
        
        self.batch_size = batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = list(set(z_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz])/self.N

        
        self.lb1 = (self.S[1,1])/(self.S[1,1]+(self.S[1,0]))
        self.lb2 = (self.S[-1,1])/(self.S[-1,1]+(self.S[-1,0]))
    
    
    def adjust_lambda(self):
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are decribed in the paper.

        """
        
        self.model.eval()
        logit = self.model(self.x_data)

        criterion = torch.nn.BCELoss(reduction = 'none')
        
                
        if self.fairness_type == 'eqopp':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit)+1)/2, (self.y_data+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                self.lb1 += self.alpha
            else:
                self.lb1 -= self.alpha
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1 
                
        elif self.fairness_type == 'eqodds':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit)+1)/2, (self.y_data+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:
                    self.lb2 += self.alpha
                else:
                    self.lb2 -= self.alpha
                    
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1
                
        elif self.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array)
            dp_loss = criterion((F.tanh(logit)+1)/2, ones_tensor) # Note that ones tensor puts as the true label
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.yz_index[tmp_yz]])) / self.z_len[tmp_yz[1]]
                    
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]: 
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha
                    
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1


    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index

    
    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """
        
        
        if self.fairness_type == 'original':
            
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)
            
            for i in range(self.batch_num):
                yield sort_index[i]
            
        else:
        
            self.adjust_lambda() # Adjust the lambda values
            each_size = {}
            
            
            # Based on the updated lambdas, determine the size of each class in a batch
            if self.fairness_type == 'eqopp':
                # lb1 * loss_z1 + (1-lb1) * loss_z0
                
                each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                each_size[(-1,1)] = round(self.S[(-1,1)])
                each_size[(-1,0)] = round(self.S[(-1,0)])
                
            elif self.fairness_type == 'eqodds':
                # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
                each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))
                
            elif self.fairness_type == 'dp':
                # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
                each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))


            # Get the indices for each class
            sort_index_y_1_z_1 = self.select_batch_replacement(each_size[(1, 1)], self.yz_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(-1, 1)], self.yz_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.yz_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(-1, 0)], self.yz_index[(-1,0)], self.batch_num, self.replacement)
            
                
            for i in range(self.batch_num):
                key_in_fairbatch = sort_index_y_0_z_0[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))
                             
                random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch
                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)


class FairBatchMultilabel(Sampler):
    """FairBatch for Multilabel Multiclass Classification.
    
    Note: This is a modified example demonstrating how to adapt the original 
    single-label FairBatch sampler to a multi-label scenario. The fairness logic 
    (eqopp, eqodds, dp) may need deeper theoretical adjustments for multi-label 
    tasks. Here we focus on code correctness and basic adaptation.
    """

    def __init__(self, model, x_tensor, y_tensor, z_tensor, batch_size, alpha, target_fairness, replacement=False, seed=0):
        self.model = model

        np.random.seed(seed)
        random.seed(seed)

        self.x_data = x_tensor
        self.y_data = y_tensor  # N x L (multi-label), each entry is 0 or 1
        self.z_data = z_tensor  # N-dim

        self.alpha = alpha
        self.fairness_type = target_fairness
        self.replacement = replacement

        self.N = len(z_tensor)
        self.batch_size = batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)

        # For multi-label, let y_item be the label indices [0, 1, 2, ..., L-1]
        self.y_item = list(range(self.y_data.shape[1]))
        self.z_item = list(set(z_tensor.tolist()))

        # Create combinations for (label_index, z_value)
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))

        # Create masks and indices for (label_index, z_value)
        self.yz_mask = {}
        self.yz_index = {}
        self.yz_len = {}

        for (lbl_idx, z_val) in self.yz_tuple:
            mask = (self.y_data[:, lbl_idx] == 1) & (self.z_data == z_val)
            self.yz_mask[(lbl_idx, z_val)] = mask
            self.yz_index[(lbl_idx, z_val)] = torch.where(mask)[0]
            self.yz_len[(lbl_idx, z_val)] = len(self.yz_index[(lbl_idx, z_val)])

        # Default batch size proportion for each yz group
        self.S = {}
        for yz in self.yz_tuple:
            # Avoid division by zero if self.N == 0
            if self.N > 0:
                self.S[yz] = self.batch_size * (self.yz_len[yz] / self.N)
            else:
                self.S[yz] = 0

        # Initialize lambda for each yz group
        total_S = sum(self.S.values()) if sum(self.S.values()) > 0 else 1
        self.lb = {yz: (self.S[yz] / total_S) for yz in self.yz_tuple}

    def adjust_lambda(self):
        """Adjust lambda values for fairness."""
        self.model.eval()
        with torch.no_grad():
            logit = self.model(self.x_data)
        
        criterion = torch.nn.BCELoss(reduction='none')

        # Compute loss per yz group (for the specific label)
        loss_per_yz = {}
        for (lbl_idx, z_val) in self.yz_tuple:
            yz_indices = self.yz_index[(lbl_idx, z_val)]
            if len(yz_indices) > 0:
                # Compute loss only for that label dimension
                pred = torch.sigmoid(logit[yz_indices, lbl_idx])
                true = self.y_data[yz_indices, lbl_idx].float()
                loss_val = criterion(pred, true)  # This is 1D now
                loss_per_yz[(lbl_idx, z_val)] = float(torch.mean(loss_val))
            else:
                loss_per_yz[(lbl_idx, z_val)] = 0.0

        # Adjust lambda values based on fairness type
        # Note: This is a simplified approach. The original code handles eqopp, eqodds, dp 
        # differently for binary single-label tasks. For multi-label scenario, one might 
        # need to define fairness constraints more carefully. Here, we do a simple approach:
        if self.fairness_type in ['dp', 'eqopp', 'eqodds']:
            # For demonstration: increase lambda of a group if its loss is higher than others
            for yz in self.yz_tuple:
                others = [other for other in self.yz_tuple if other != yz]
                if len(others) > 0:
                    max_loss = max(loss_per_yz[other] for other in others)
                    if loss_per_yz[yz] > max_loss:
                        self.lb[yz] += self.alpha
                    else:
                        self.lb[yz] -= self.alpha

                    # Ensure lambda is within [0, 1]
                    self.lb[yz] = max(0, min(1, self.lb[yz]))

    def select_batch(self, size, indices, replacement=False):
        """Select a batch of indices."""
        indices = np.array(indices)
        if len(indices) == 0:
            return []
        if replacement:
            return np.random.choice(indices, size, replace=True).tolist()
        else:
            if size <= len(indices):
                chosen = np.random.choice(indices, size, replace=False).tolist()
            else:
                # If requested size > available, cycle through indices
                chosen = np.random.choice(indices, size, replace=True).tolist()
            return chosen

    def __iter__(self):
        """Generate batches."""
        if self.fairness_type == 'original':
            all_indices = np.arange(len(self.y_data))
            # Shuffle once
            np.random.shuffle(all_indices)
            for i in range(self.batch_num):
                yield all_indices[i * self.batch_size:(i + 1) * self.batch_size]
        else:
            self.adjust_lambda()

            # Recompute total mass to properly scale batch sizes
            total_lb = sum(self.lb.values()) if sum(self.lb.values()) > 0 else 1
            batch_indices = []
            for yz in self.yz_tuple:
                yz_size = int(round((self.lb[yz] / total_lb) * self.batch_size))
                batch_indices.extend(self.select_batch(yz_size, self.yz_index[yz].tolist(), self.replacement))

            # If due to rounding we don't get enough samples, fill the rest randomly
            if len(batch_indices) < self.batch_size:
                all_indices = np.arange(len(self.y_data))
                needed = self.batch_size - len(batch_indices)
                batch_indices.extend(self.select_batch(needed, all_indices, self.replacement))

            random.shuffle(batch_indices)
            for i in range(self.batch_num):
                start = i * self.batch_size
                end = start + self.batch_size
                if end > len(batch_indices):
                    # wrap around or re-sample if needed
                    all_indices = np.arange(len(self.y_data))
                    needed = end - len(batch_indices)
                    batch_indices.extend(self.select_batch(needed, all_indices, self.replacement))
                yield batch_indices[start:end]

    def __len__(self):
        return len(self.y_data)
