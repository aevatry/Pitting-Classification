import torch
import numpy as np
import os
from torch.utils.data import Dataset

class LONG_NORM (Dataset):

    def __init__(self, train_dir:str,  **kwargs):
        super().__init__()

        options = ['min_len', 'max_start_index']
        default = [3600, 5000]
        self.get_opts(options, default, kwargs)

        # Checking for different types of text based files
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
        return max(self.all_series_len) - self.min_len

    def __getitem__(self, idx):

       # Draw the random sequence lenght, centered around mid sequence lenght (z=0.5)
       a = 0.5
       z = np.random.uniform(size = 1)
       z = a*z[0]**3 - a*1.5*z[0]**2 + (1+0.5*a)*z[0]
       sequence_lenght = int(self.min_len + z*(max(self.all_series_len) - self.min_len))

       existence = self._selfbatch(self.all_series_len, sequence_lenght)


       fail_class = np.array([[self.series_t[-1][sequence_lenght]] for i in existence])
       label = self.get_label(fail_class)

       # List of all series features
       features = np.array([get_R(self.series_t[i][:-1, 0 : sequence_lenght], ma_win=300)  for i in existence])

       features = torch.as_tensor( features.squeeze(2), dtype= torch.float32)

       return features, label


    def _selfbatch(self, all_series_len, seq_len):
        # decides which series are included in thos batch
        existence = [i for i in range(len(all_series_len)) if seq_len<all_series_len[i]]
        return existence

    def get_label(self, fail_class):

        labels = np.zeros((fail_class.shape[0:-1], ))

        return labels

    def get_opts(self,options:list, defaults:list, kwargs: dict)-> None:

        assert len(options) == len(defaults), f"The defaults need to match the options but get lenght {len(options)} and {len(defaults)}"

        for i, opt in enumerate(options):
            opt_val = kwargs.get(opt, None)

            if opt_val == None:
                opt_val = defaults[i]
                print(f"Option {opt} set to default: {opt_val}")
            else:
                print(f"Option {opt} set to custom: {opt_val}")

            setattr(self, opt, opt_val)






######## Helper functions ########

def EMA(sequence: np.ndarray, ma_win: int):
    """
    Exponential moving average of time series

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght)
        ma_win (int): number of points taken in the moving average window

    Returns:
        Sk (np.ndarray): array of the corresponding moving average values
    """
    EMA_vals = np.empty((sequence.shape[0], sequence.shape[1] - ma_win + 1))
    alpha = 2/(ma_win+1)

    for i in range(0, sequence.shape[1] - ma_win + 1):
        if i == 0:
            EMA_vals[:,0] = (1/ma_win)*np.sum(sequence[:, 0 : 0+ma_win], axis = 1)

        else:
            EMA_vals [:,i] = (1-alpha)*EMA_vals[:,i-1] + alpha*sequence[:, i+ma_win-1]

    return EMA_vals

def SMA(sequence: np.ndarray, ma_win: int):
    """
    Simple moving average of time series

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght)
        ma_win (int): number of points taken in the moving average window

    Returns:
        Sk (np.ndarray): array of the corresponding moving average values
    """
    SMA_vals = np.empty((sequence.shape[0], sequence.shape[1] - ma_win + 1))

    for i in range(0, sequence.shape[1] - ma_win + 1):

        SMA_vals[:,i] = (1/ma_win)*np.sum(sequence[:, i : i+ma_win], axis = 1)

    return SMA_vals


def get_R(sequence: np.ndarray, ma_win:int, high_norma:int = 1, low_norma:int = 0, MA_mode:str = 'EMA', sl_win:int = None):

    """
    Get the Adaptive Normalization of the input sequence

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght). Lenght need to be the same for all features
        ma_win (int): number of points taken in the moving average window
        high_norma (int, default = 1): higher value for normalisation
        low_norma (int, default = 0): lower value for normalisation
        MA_mode (str, default = 'EMA): Mode of the moving average series. Two implemented: 'EMA' (Exponential moving average) and 'SMA' (Simple Moving Average)
        sl_win (int, optional): number fo points in each Disjoint Sliding Window. Default is put to lenght of the sequence

    Returns:
        R (np.ndarray): Normalized sequence with dimensions (batch, features, lenght) if sl_win = lenght
                        IF sl_win != lenght, we have R (batch, feature, lenght - sl_win +1, sl_win)
    """

    if MA_mode == 'SMA':
        MA = SMA(sequence, ma_win)
    elif MA_mode == 'EMA':
        MA = EMA(sequence, ma_win)
    else:
        raise NotImplementedError(f"The moving average mode needs to be 'EMA' or 'SMA' but got {MA_mode} instead")

    if sl_win == None:
        sl_win = sequence.shape[-1]

    R = np.zeros((sequence.shape[0], sequence.shape[1] - sl_win + 1, sl_win))

    for i in range(0, R.shape[1]):
        S = sequence[:,i:i+sl_win]
        Sk = MA[:,i].reshape((R.shape[0],-1))
        R_new = S/Sk
        R[:,i] = R_new

    _outliers_norm(R, high_norma, low_norma)
    return R



def _outliers_norm (R, high_norma, low_norma):

    quantiles = np.array([np.quantile(arr, [0.25,0.75], axis = 0) for arr in R.reshape((R.shape[0], -1))])

    IQR = (quantiles[:, 1] - quantiles[:, 0]).reshape((R.shape[0], -1))

    low_lim = quantiles[:,0].reshape((R.shape[0], -1)) - 1.5*IQR
    high_lim = quantiles[:,1].reshape((R.shape[0], -1)) + 1.5*IQR

    # set rows/Disjoint Sliding Window to NONE if any value in row outside of quartiles
    # block below is NOT to be used: because we want really long w sections in the R series, really high probability that one of the vals will be off: voids to whole line. And we still need corrects low_lim and high_lim for minmax normalization
    '''
    for i,batch in enumerate(R):
        for j,feature in enumerate(batch):
            for m, DSW in enumerate(feature):

                cond_list = [True for val in DSW if val<lims[i][j][0] or val>lims[i][j][1]]

                if True in cond_list:
                    R[i,j,m].fill(None)
    '''

    _minmax_norm(R, high_lim, low_lim, high_norma, low_norma)

def _minmax_norm(R, high_lim, low_lim, high_norma, low_norma):

    min_R = np.min( R.reshape((R.shape[0], -1)), axis=1 ).reshape((R.shape[0], -1))
    max_R = np.max( R.reshape((R.shape[0], -1)), axis=1 ).reshape((R.shape[0], -1))

    min_a = np.max(np.concatenate((min_R, low_lim), axis = 1), axis = 1).reshape(R.shape[0], -1)
    max_a = np.min(np.concatenate((max_R, high_lim), axis = 1), axis = 1).reshape(R.shape[0], -1)

    for i in range(0,R.shape[-2]):
        for j in range(0, R.shape[-1]):

            holder = (high_norma - low_norma)* (R[:, i,j].reshape((R.shape[0], -1)) - min_a)/(max_a - min_a) + low_norma
            R[:, i, j] = holder.reshape((R.shape[0]))
