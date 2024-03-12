import torch
import numpy as np
import os
from torch.utils.data import Dataset
class LONG_NORM (Dataset):

    """
    PyTorch Dataset to extract times series features and labels. It goes through the whole range of all series in steps and extract the corresponding features. The series need to be organised as rows of n data points (each row is a feature) and the labels are the last row. The start point for the sequence is decided using random distribution in a specified interval. This Dataset class self batches, so batch need to be set to 1 in a DataLoader. It batches with all available series and sub-batches with different starting points within each serie. 

    Args:
        train_dir (str): Directory where all files (each file is 1 run) are stored
        **kwargs (dict): Dictionnary to pass additional optional parameters
            Optional parameters:
                min_length (int = 5000): minimum length of extracted features
                max_start (int = 3600): max starting index to draw the starting points
                sub_batches (int = 5): number of sub-batches to extract for each time series. Minimum 1
                sequence_step (int = 5): Factor by which the size of the dataset is reduced per epoch. Minimum 1
                dmg_start (float = 0.0): Float between 0 and 1 for the initial proportion of sequence that is considered non-damage
    """

    def __init__(self, train_dir:str,  **kwargs):
        super().__init__()

        options = ['min_length', 'max_start', 'sub_batches', 'sequence_step', 'dmg_start']
        default = [5000, 3600, 5, 5, 0.]
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
        # get length -1 because we want to vary the lenght at each sequence lenght by +/- sequence_step (see __getitem__)
        return int((max(self.all_series_len) - self.min_length) / self.sequence_step) - 1

    def __getitem__(self, idx):
        
        # We still get random lengths even if we go through the whole dataset in steps
        sequence_length = int(self.min_length + self.sequence_step*idx) + np.random.randint(low=-self.sequence_step, high=self.sequence_step+1)


        features = []
        labels = []

        for idx, timeserie in enumerate(self.series_t):

            start_point_range = self.all_series_len[idx] - sequence_length
            starting_points = self.get_starting_points(start_point_range)

            # verify that starting points exist
            if isinstance(starting_points, np.ndarray):
        
                for start_idx in starting_points:
                    # extract features and labels
                    # add -1 to the sequence lenght to adapt to error: index n out-of-bounds for axis with size n
                    # worst case: random-sequence-length = len(sequence). So for that, we use sequence_length - 1
                    fail_class = timeserie[-1][sequence_length-1 + start_idx]

                    # For Pitting ONLY: only have damage label if drawn sequence stops before a custom set point
                    if fail_class==2 and sequence_length+start_idx < self.dmg_start*self.all_series_len[idx]:
                        fail_class=0

                    labels += [fail_class]
                    features += [get_R(timeserie[:-1, start_idx : sequence_length-1 + start_idx], ma_win=300)]
        
        labels = np.array(labels, dtype=int)
        features = torch.as_tensor(np.array(features), dtype= torch.float32)
        return features, labels


    def get_starting_points (self, start_point_range: int):

        """
        Draws random starting points based on the max available rang we can choose from. 

        Returns:
            List : if starting points exist, 
            None : otherwise

        """

        # Logic to choose the starting points
        if start_point_range > self.max_start:
            # draw geometric up to max starting range if start point range is too big
            # the p parameter was chosen based on testing
            start_points = np.random.geometric(p = 8/self.max_start, size = self.sub_batches)
            start_points[np.where(start_points>self.max_start)] = self.max_start

        elif self.max_start >= start_point_range > 80:
            # draw geometric up to start point range if it is small enough
            # 80 is chosen based on 8 (only draw geometric if p<0.1)
            start_points = np.random.geometric(p = 8/start_point_range, size = self.sub_batches)
            start_points[np.where(start_points>start_point_range)] = start_point_range

        elif 80 >= start_point_range >= 0:
            start_points = np.random.choice(start_point_range+1, size= min(self.sub_batches, start_point_range+1),replace=False)


        else:
            start_points = None

        return start_points
    

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