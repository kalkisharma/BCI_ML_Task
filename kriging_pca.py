import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

# We do this to ignore several specific Pandas warnings
import warnings
warnings.filterwarnings("ignore")

# location of raw data folder and param file
RAW_DATA_PATH = "./simulation_files"
PARAM_PATH = "./case_parameters_sim.csv"

# create X and Y datasets from the raw data files and the parameter file
class Dataset:

    def __init__(self, RAW_DATA_PATH, PARAM_PATH):
        self.raw_path = RAW_DATA_PATH
        self.param_path = PARAM_PATH

        self.create_dataset()

    def create_dataset(self):

        parameters = []
        # read parameters
        with open(self.param_path) as f:
            csv_file = csv.reader(f)
            next(csv_file) # skip column names
            for r in csv_file:
                if r[12] == 'Y' and float(r[1]) != 5 and abs(float(r[5])) != 4 and float(r[3]) != 0.1 and float(r[4]) != 0.1 and float(r[3]) != 0.8 and float(r[4]) != 0.8:
                    parameters.append([float(i) for i in r[0:11]])
                                        

        self.dataX = np.empty((len(parameters), 3))
        for i,param in enumerate(parameters):
            
            # open the data file
            path = f'{self.raw_path}/RUN_{int(param[0])}'
            filename = path + '/2D_FZ_FORMATTED.txt'
            _,y1,y2 = pickle.load(open(filename, 'rb'))  # x values, y1 values, y2 values

            # store param data
            self.dataX[i] = param[3:6]

            #data stored as single element tuples which makes it hard to see the 2d shape 
            # so converting it to standard 1d array
            
            # so we can handle an unknown data lengths
            if i == 0:
                self.dataY1 = np.zeros((len(parameters),len(y1)))
                self.dataY2 = np.zeros((len(parameters),len(y2)))
            
            # some of them have different number of points - ask kalki about this
            self.dataY1[i,0:len(y1)] = y1
            self.dataY2[i,0:len(y2)] = y2
        
        mach_left = np.asarray(self.dataX[:, 0], dtype=float)
        mach_right = np.asarray(self.dataX[:, 1], dtype=float)
        sepa = self.dataX[:, 2]
        
        for data in self.dataX:
            if data[-1] == 1:
                plt.scatter(data[1], data[0], color='k')
        plt.title(r'$\frac{\Delta z}{chord}=1$', fontsize=20)
        plt.xlabel(r'$M_{RIGHT}$', fontsize=14)
        plt.ylabel(r'$M_{LEFT}$', fontsize=14)
        plt.xlim(0.1, 0.9)
        plt.ylim(0.1, 0.9)
        plt.grid()
        plt.show()
        

        for data in self.dataX:
            if data[-1] == 2:
                plt.scatter(data[1], data[0], color='k')
        plt.title(r'$\frac{\Delta z}{chord}=2$', fontsize=20)
        plt.xlabel(r'$M_{RIGHT}$', fontsize=14)
        plt.ylabel(r'$M_{LEFT}$', fontsize=14)
        plt.xlim(0.1, 0.9)
        plt.ylim(0.1, 0.9)
        plt.grid()
        plt.show()
        input('..')
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


#----------------------------------------------------
#----------CREATE DATASET----------------------------
#----------------------------------------------------

# create the dataset from files    
dataset = Dataset(RAW_DATA_PATH, PARAM_PATH)

#test train split to prevent cheating - t for train, T for Test
# using indices to prevent multiple copies of data and since we have multiple Ys
indices = list(range(len(dataset.dataY1)))
dataX_t, dataX_T, idx_t, idx_T = train_test_split(dataset.dataX, indices, test_size = 0.3)

# just to make it easier to type
dataY1_t = dataset.dataY1[idx_t]
dataY2_t = dataset.dataY2[idx_t]
dataY1_tt = np.copy(dataY1_t)

print(f"Dataset contains {dataset.dataX.shape[0]} samples of length {dataset.dataY1.shape[1]}")
print(f"\tTrainset of {dataX_t.shape[0]} samples and test set of {dataX_T.shape[0]} samples ({dataX_T.shape[0]/dataset.dataX.shape[0]*100:2.0f}%)")


#----------------------------------------------------
#----------PRE-PROCESSING----------------------------
#----------------------------------------------------
# set true those you want to use

preprocess_dict = {'normalize':True, 'standardize':False, 'pca':True}

print("Preprocessing steps:")
for k in preprocess_dict.keys():
    if preprocess_dict[k]:
        print("\t",k)

#-------------------------------
# min max normalize data

if preprocess_dict['normalize']:
    normalizer_y1 = NormalizeMinMaxRowwise()
    normalizer_y2 = NormalizeMinMaxRowwise()

    normalizer_y1.fit(dataY1_t)
    normalizer_y2.fit(dataY2_t)

    dataY1_t = normalizer_y1.transform(dataY1_t)
    dataY2_t = normalizer_y1.transform(dataY2_t)
    
    
    # sanity check the graphs 
    """
    dataY1_t_ninv = normalizer_y1.inverse_transform(dataY1_t)
    for i in range(dataY1_t.shape[0]):
        fig, ax = plt.subplots(3)
        ax[0].plot(dataY1_t[i])
        ax[1].plot(dataY1_t_ninv[i])
        ax[2].plot(dataset.dataY1[idx_t][i])
        plt.show()
    """

#-------------------------------
# scale data

if preprocess_dict['standardize']:
    scaler_y1 = StandardScaler().fit(dataY1_t)
    scaler_y2 = StandardScaler().fit(dataY2_t)
    
    dataY1_t = scaler_y1.transform(dataY1_t)
    dataY2_t = scaler_y2.transform(dataY2_t)

    # sanity check the graphs 
    """
    dataY1_t_sinv = scaler_y1.inverse_transform(dataY1_t)
    for i in range(dataY1_t.shape[0]):
        fig, ax = plt.subplots(3)
        ax[0].plot(dataY1_t[i])
        ax[1].plot(dataY1_t_sinv[i])
        ax[2].plot(dataset.dataY1[idx_t][i])
        plt.show()
    """

#-------------------------------
# PCA

if preprocess_dict['pca']:
    pca_y1 = PCA(n_components=0.999)
    pca_y2 = PCA(n_components=0.999)
    
    pca_y1.fit(dataY1_t)
    pca_y2.fit(dataY2_t)

    dataY1_t = pca_y1.transform(dataY1_t)
    dataY2_t = pca_y2.transform(dataY2_t)

    # sanity check
    """
    dataY1_t_pinv = pca_y1.inverse_transform(dataY1_t)
    for i in range(dataY1_t.shape[0]):
        fig, ax = plt.subplots(3)
        ax[0].plot(dataY1_t[i])
        ax[1].plot(dataY1_t_pinv[i])
        ax[2].plot(dataset.dataY1[idx_t][i])
        plt.show()
    exit()
    """

#----------------------------------------------------
#----------TRAIN & VALIDATE--------------------------
#----------------------------------------------------

print("Training GPR: ")

# Instantiate a Gaussian Process model
kernel = ConstantKernel(1.0, (1e-1, 1e1)) * RBF([1.0]*dataX_t.shape[1], (1e-2, 1e2)) # mse - 0.016, 0.017(pca), 0.004(norm+pca), 0.007(norm+scale+pca)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=30)

import time
t1 = time.time()
gp.fit(dataX_t,dataY1_t)
print(f"\tTook {time.time()-t1:.2f} seconds")

# predict
y_pred, sigma = gp.predict(dataX_t,return_std=True)

fig, ax = plt.subplots(4, 4, sharex=True, figsize=(30,30))
dx = np.linspace(-30, 30, 6001)
for i in range(16):

    # inverse preprocess
    if preprocess_dict["pca"]:
        y = pca_y1.inverse_transform(y_pred[i]) # inverse pca
    if preprocess_dict["standardize"]:
        y = scaler_y1.inverse_transform(y)
    if preprocess_dict["normalize"]:
        y = normalizer_y1.inverse_transform(y)

    # graph the results
    
    plt.sca(ax[i//4, i%4]) 
    plt.title(dataX_t[i])
    plt.plot(dx, dataY1_tt[i], label='SIMULATION', linewidth=1.5, color='blue', alpha=0.8)
    plt.plot(dx, y, label='PREDICTION', linewidth=2, color='black', linestyle=':')
    plt.xlim(-30, 30)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.grid(which='major')
    plt.legend()
    #ax.fill(np.concatenate([x_func, x_func[::-1]]), np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.3, fc='cyan', ec='None', label='95% confidence interval')
    #plt.show()

fig.suptitle('TRAINING SET')
plt.show()

print("Testing Model")

# predict
y_pred, sigma = gp.predict(dataX_T,return_std=True)

# calculate error and visualize
dataY1_T = dataset.dataY1[idx_T] # ground truth
dataY2_T = dataset.dataY2[idx_T] # ground truth

total_mse = np.zeros(y_pred.shape[0])

fig, ax = plt.subplots(4, 4, sharex=True, figsize=(30,30))
dx = np.linspace(-30, 30, 6001)
count = 0
for i in range(y_pred.shape[0]):

    # inverse preprocess
    if preprocess_dict["pca"]:
        y = pca_y1.inverse_transform(y_pred[i]) # inverse pca
    if preprocess_dict["standardize"]:
        y = scaler_y1.inverse_transform(y)
    if preprocess_dict["normalize"]:
        y = normalizer_y1.inverse_transform(y)

    # normalize and calc error - normalized by range of sample to help comparison
    maximum = dataY1_T[i].max()
    minimum = dataY1_T[i].min()
    y2 = dataY1_T[i]
    y2 = (y2 - minimum)/(maximum-minimum)
    y1 = y
    y1 = (y1 - minimum)/(maximum-minimum)

    mse = (np.square(y1 - y2)).mean(axis=0)
    total_mse[i]= mse
    """ 
    if mse > 0.01:
        print(f"{i} ({mse}): ", dataX_T[i])
        #fig, ax = plt.subplots(2)
        #ax[0].plot(dataY1_T[i])
        #ax[1].plot(y)
        #plt.show()
    """
    #if mse > 0.01:
    #    continue
    # graph the results
    
    plt.sca(ax[count//4, count%4]) 
    plt.title(dataX_T[i])
    plt.plot(dx, dataY1_T[i], label='SIMULATION', linewidth=1.5, color='blue', alpha=0.8)
    plt.plot(dx, y, label='PREDICTION', linewidth=2, color='black', linestyle=':')
    plt.xlim(-30, 30)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.grid(which='major')
    plt.legend()
    count += 1
    if count == 16:
        break
    #ax.fill(np.concatenate([x_func, x_func[::-1]]), np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.3, fc='cyan', ec='None', label='95% confidence interval')
    #print(mse)
    #plt.show()
    
fig.suptitle('TESTING SET')
plt.show()

# some stats
print("\tMSE:")
print("\t\tmean: ", np.mean(total_mse))
print("\t\tmedian: ", np.median(total_mse))
print("\t\tmax: ", np.max(total_mse))

