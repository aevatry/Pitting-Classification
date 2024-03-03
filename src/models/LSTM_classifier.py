from torch import nn
import numpy as np
import torch

class LSTM_classifier(nn.Module):

    def __init__(self, **kwargs): 
        """
        Arguments:
            kwargs (unpacked dict, optional): dict to set the options, where the keys and explanation of options are:
                _pyramid_bins : list (default = [300, 100]) -> bins to be used for pyramid pooling operation
                _pooling_mode : str (default = 'max') -> Choose between implemented pooling modes 
                _num_feats: int (required) -> how many features in input dataset
        """

        super().__init__()
        try:
            self._pyramid_bins = kwargs['_pyramid_bins']
            print(f"pyramid pooling bins set to custom value: {self._pyramid_bins}")
        except:
            self._pyramid_bins = [300, 100]
            print(f"pyramid pooling bins set to default: {self._pyramid_bins}")

        try:
            self._pooling_mode = kwargs['_pooling_mode']
            print(f"pooling mode set to custom value: {self._pooling_mode}")
        except:
            self._pooling_mode = 'max'
            print(f"pooling mode set to default: {self._pooling_mode}")

        try:
            self._num_feats = kwargs['_num_feats']
            print(f"number of features: {self._num_feats}")
        except:
            raise ValueError("You need to specify the number of features")
        
 
        self.conv1_1D_depthwise = nn.Conv1d(in_channels= self._num_feats, out_channels= self._num_feats*16, kernel_size= 30, stride= 1, groups= self._num_feats) #depthwise
        self.conv1_1D_pairwise = nn.Conv1d(in_channels= self._num_feats*16, out_channels= 32, kernel_size= 1, stride= 1) #pairwise
        
        self.conv2_1D = nn.Conv1d(in_channels = 32, out_channels= 64, kernel_size= 10, stride = 1)

        # Instantiation of pooling with linear block
        self.spatial_pool_block = Spatial_LSTM_block(pyramid_bins=self._pyramid_bins, entry_channels= 64, _mode=self._pooling_mode)

        # Number of input features: sum of all bin sizes
        self.classify_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(len(self._pyramid_bins), 1))

        self.LeakyReLu = nn.LeakyReLU()
        self.logmax = nn.LogSoftmax(dim=1)

    def forward (self, X): #This function is the one called to do a forward pass
        
        # First Separable 1D Convolution
        X = self.conv1_1D_depthwise(X)
        X = self.conv1_1D_pairwise(X)
        X = self.LeakyReLu(X)

        # Variational Pooling
        X = var_pool_func(previous_conv=X, _mode = self._pooling_mode)

        # Second 1D Convolution
        X = self.conv2_1D(X)
        X = self.LeakyReLu(X)

        # Spatial Pyramidal Pooling Block
        X = self.spatial_pool_block(X) 

        X = torch.unsqueeze(X, 1)
        # Fully Connected Layers
        X = self.classify_conv(X)

        X = torch.squeeze(X)
        X = self.logmax(X)
        return X
    
    def get_total_bins(self):
        
        list_to_array = np.array(self._pyramid_bins)
        assert len(np.array(list_to_array).shape) == 1, 'Bins should be 1 dimensional'
        return np.sum(list_to_array, axis = 0)
    


class Spatial_LSTM_block (nn.Module):

    """
    Spatial pyramidal pool class implemented with mps compatibility (with reduced functionality like no dynamic kernel size and stride). 
    Structure used : https://github.com/addisonklinke/pytorch-architectures/blob/master/torcharch/conv.py

    Args:
        pyramid_bins (list): list of all output size for eacg convolution
        entry_shape (list): shape of entry convolution
        _mode (str): convolution mode. implemented are 'max' for maxpool and 'avg' for average pool

    When forward method is called, accepts a tensor X of shape (batch, features, input_lenght)
    Returns a tensor of shape (batch, features, sum(bins)) where each bin pooling operation is concatenated in the 2nd dimension
    """

    # not need to pass the previous convolution in the __init__ because it is pass in the forward pass
    def __init__(self, pyramid_bins:list, entry_channels:int, _mode:str):
        super().__init__()

        self.name = 'Spatial_LSTM_block' 

        self.pyramid_bins = pyramid_bins
        self._mode = _mode

        # Conv layer params
        self.conv_filters = 4

        # LSTM hidden layers
        self.hidden_layers = 64

        
        #nn.ModuleList allows PyTorch to find the convolution layers
        self.spatial_conv = nn.ModuleList([])
        self.LSTMcell = nn.ModuleList([])
        self.LSTMlin = nn.ModuleList([])

        for _ in pyramid_bins:

            self.spatial_conv.append(nn.Sequential(
                nn.Conv1d(in_channels=entry_channels, out_channels=self.conv_filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            ))

            self.LSTMcell.append(nn.LSTMCell(input_size=self.conv_filters, hidden_size= self.hidden_layers))

            self.LSTMlin.append(nn.Linear(in_features=self.hidden_layers, out_features=3))


    
    def forward(self, X):
        
        # X is the result of the previous convolution
        assert X.dim() == 3, 'Expect a 3D input: (Batch, Channels, Lenght)'


        out =[]
        for i, bin_size in enumerate(self.pyramid_bins):

            
            #pipe each output of the pyramid pooling to the 1D CNN
            X_pyr = self.get_adaptive_pool(X, bin_size)
            X_conv = self.spatial_conv[i](X_pyr)

            # initial hidden and cell states
            h_t = torch.zeros(X_conv.shape[0], self.hidden_layers)
            c_t = torch.zeros(X_conv.shape[0], self.hidden_layers)

            for index in range(X_conv.shape[-1]):

                conv_elmt = X_conv[:,:, index]
                h_t, c_t = self.LSTMcell[i](conv_elmt, (h_t, c_t)) 
            
            Xout = self.LSTMlin[i](h_t)
            out+=[Xout]
            
        return torch.stack(out, dim=1)
    

    def get_adaptive_pool (self, previous_conv:torch.Tensor, output_size:int)-> torch.Tensor:

        if self._mode == 'max':
            pool_func = torch.nn.functional.max_pool1d
        elif self._mode == 'avg':
            pool_func = torch.nn.functional.avg_pool1d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{self._mode}', expected 'max' or 'avg'")

        stride = (previous_conv.shape[2]//output_size) # floor division
        kernel_size = previous_conv.shape[2] - (output_size-1)*stride

        return pool_func(previous_conv, kernel_size, stride)
    




def var_pool_func(previous_conv: torch.Tensor, kernel_size: int=20, stride: int=5, _mode = 'max') -> torch.Tensor:

    '''
    Performs a pooling, whose mode is decided by the _mode variable, using the functional API of PyTorch

    Args:
    
        previous_conv (torch.Tensor): Tensor of the output of the previous convolution layer
        kernel_size (int): desired kernel size
        stride (int): desired stride
        _mode (str): accepts 'max' for max pooling or 'avg' for average pooling
    
    Returns: 
        Output of torch.nn.functional.max_pool1d or torch.nn.functional.avg_pool1d depending on mode with ceil=true for the specified parameters of the inputs

    '''  

    # Get the right pooling function
    if _mode == 'max':
        pool_func = torch.nn.functional.max_pool1d
    elif _mode == 'avg':
        pool_func = torch.nn.functional.avg_pool1d
    else:
        raise NotImplementedError(f"Unknown pooling mode '{_mode}', expected 'max' or 'avg'")


    return pool_func(previous_conv, kernel_size=kernel_size, stride=stride, ceil_mode=True)