import torch
import numpy as np
import os
from torch.utils.data import Dataset

class LONG_NORM_EVAL(Dataset):

    """
    Arguments:
        train_dir (string): path to a directory with csv files with series of features with labels (labels are csv last series) used for testing the model.
        step (int, default = 10): step between two consecutive sequence lenght
        file_num (integer, default = 1): in range [1; k] with k the number of files in the directory. Default 1
        min_lenght (int, default = 5000): The minimum lenght for all series
    """

    def __init__(self, train_dir:str, step:int = 10, file_num:int =1, min_length:int = 3600):


        self.step = step
        self.file_num = file_num - 1
        self.min_length = min_length

        # Checking for different types 
        signs = [' ', ',', ';','    ']
        series_t = []
        for file_path in os.listdir(train_dir):

            full_path = '/'.join([train_dir, file_path])

            for sign in signs:
                try:
                    series_t += [np.loadtxt(full_path, delimiter=sign)]
                    print(f"Sign : '{sign}' works")
                except : 
                    pass

        self.series_t = series_t
        all_series_len = [series_t[i].shape[1] for i in range(len(series_t))]
        self.all_series_len = all_series_len

    def __len__(self):
        return int((self.all_series_len[self.file_num] - self.min_length) / self.step) - 1
    
    
    def __getitem__(self, idx):
        
        sequence_lenght = int(self.min_length + self.step*idx)

        labels = []
        features = []
        # Extract current label (1 sample)
        fail_class = self.series_t[self.file_num][-1, sequence_lenght]
        labels += [fail_class]
        # Extract series features
        features += [get_R(self.series_t[self.file_num][:-1, 0 : sequence_lenght], ma_win=300)]


        labels = np.array(labels, dtype=int)
        features = torch.as_tensor(np.array(features), dtype= torch.float32)

        return features, labels
    



######## Helper functions ########


def get_R (sequence: np.ndarray, ma_win:int, high_norma:int = 1, low_norma:int = 0):

    """
    Get the Adaptive Normalization of the input sequence

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght). Lenght need to be the same for all features
        ma_win (int): number of points taken in the moving average window
        high_norma (int, default = 1): higher value for normalisation
        low_norma (int, default = 0): lower value for normalisation

    Returns:
        R_scaled (np.ndarray): Normalized sequence with dimensions (features, lenght)
    """

    MA = get_MA(sequence, ma_win).reshape(sequence.shape[0], 1)
    R_scaled = sequence/MA

    R = normalise_R(R_scaled, high_norma, low_norma)

    return R

def get_MA(sequence, ma_win):

    MA = np.sum(sequence[:, :ma_win], axis=1)/ma_win

    return MA

def normalise_R (R_scaled, high_norma, low_norma):


    # Finding the normalization parameters
    quantiles = np.array([np.quantile(arr, [0.25,0.75]) for arr in R_scaled])

    IQR = (quantiles[:, 1] - quantiles[:, 0])
    
    low_lim = quantiles[:,0] - 1.5*IQR   
    high_lim = quantiles[:,1] + 1.5*IQR
    
    min_R = np.min( R_scaled, axis=1 )
    max_R = np.max( R_scaled, axis=1 )
    min_a = np.max(np.stack((min_R, low_lim)), axis = 0)
    max_a = np.min(np.stack((max_R, high_lim)), axis = 0)

    # Actual normalization

    a_scale = (max_a - min_a).reshape(R_scaled.shape[0], -1)
    R = (high_norma-low_norma)*(R_scaled - min_a.reshape(R_scaled.shape[0], -1))/a_scale + low_norma
    return R