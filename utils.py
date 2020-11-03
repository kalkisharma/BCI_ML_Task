import numpy as np
import matplotlib.pyplot as plt

# min max normalization to normalize over each row
class NormalizeMinMaxRowwise:
    def __init__(self):
        pass

    def fit(self, data):
        self.max_val = np.max(data)
        self.min_val = np.min(data)
    
    def transform(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, data):
        return data * (self.max_val - self.min_val) + self.min_val


def normData(x):
    _mean = np.mean(x,axis=0)
    _std = np.std(x,axis=0)
    return (x-_mean)/_std, _mean, _std

def nonDimensionalizeF(X):
    _M = X[:,0]+X[:,1]
    _param = _M**2/np.abs(X[:,2])**3/np.sqrt(1-X[:,0]**2)

    # # _param = 1 to turn off non-dimmensionalization
    # _param = np.ones(_param.shape[0])
    return _param.reshape(-1,1)

def errorCalc(pred, orig, func="nrmse"):
  
    if func == "nrmse":
        max_val = np.amax(orig,axis = 1)
        min_val = np.amin(orig, axis = 1)
        rng = (max_val - min_val)

        rmse = np.sum(np.square(orig-pred), axis = 1) / orig.shape[1]
        rmse = np.sqrt(rmse)
        nrmse = rmse/rng
        return nrmse

    if func == "rmse":
        rmse = np.sum(np.square(orig-pred), axis = 1) / orig.shape[1]
        rmse = np.sqrt(rmse)
        return rmse
    
    if func == "rl2":
        l2 = np.sqrt(np.sum(np.square( orig - pred ), axis=1))
        l2 = l2 / np.sqrt(np.sum(np.square( orig ), axis=1))
        return l2
    
    if func == "rlinf":
        linf = np.max(np.abs(orig-pred), axis=1)
        linf = linf / np.max(np.abs(orig), axis=1)
        
        return linf