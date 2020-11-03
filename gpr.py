import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split, KFold
import sklearn.decomposition as skd
from sklearn.preprocessing import StandardScaler

import oldVerGPR 
import dataset
import utils

# paths
PARAM_PATH = "../case_parameters_sim.csv"
RAW_DATA_PATH = "../simulation_files"


def gpr_new(x_train, x_test, y_train, y_test,kernel_type,err_type):

    # normalize with maximum value in dataset
    normalizer1 = utils.NormalizeMinMaxRowwise()
    normalizer1.fit(y_train)
    y_train_N = normalizer1.transform(y_train)

    # pca
    pca = skd.PCA(n_components=0.99)
    dyn_pca = pca.fit_transform(y_train_N)

    # Latest version of GPR, no trend function
    if kernel_type=='matern52':
        kernel = Matern(length_scale=.1*np.ones(x_train.shape[1]), length_scale_bounds=[1e-6,10.], nu=2.5)
    if kernel_type=='squared_exponential':
        kernel = RBF(length_scale=.1*np.ones(x_train.shape[1]), length_scale_bounds=[1e-6,10.])

    gp = GaussianProcessRegressor(kernel=kernel).fit(x_train,dyn_pca)
 
    y_predict_training = gp.predict(x_train)
    y_predict_testing = gp.predict(x_test)

    # If trained using PCA data
    y_predict_training = pca.inverse_transform(y_predict_training) 
    y_predict_testing = pca.inverse_transform(y_predict_testing) 

    y_predict_training = normalizer1.inverse_transform(y_predict_training)
    y_predict_testing = normalizer1.inverse_transform(y_predict_testing)

    val_error = utils.errorCalc(y_predict_testing,y_test, func=err_type)
    
    """
    for i in range(10):
        fig, ax = plt.subplots(2)
        ax[0].plot(y_test_o[i])
        ax[1].plot(y_predict_testing[i])

        ax[1].plot(y_test[i])
        ax[1].plot(y_predict_testing[i])
        
        plt.show()

        print ('RMSE = {0:2.5f}'.format(RMSE(y_predict_testing[i],y_test[i])))
    """

    return val_error

# with ND
#[0.04139647 0.07523885 0.03250064 0.04000943 0.04981548]
#[0.05524919 0.09066575 0.03622024 0.05267701 0.09294491]
# without ND
#[0.04541003 0.05256931 0.05199875 0.04392800 0.04181507]
#[0.04557427 0.05252665 0.05211700 0.04964280 0.04672143]

def gpr_old(xTrain, xTest, yTrain, yTest,regr_type,kernel_type,err_type):

    NDTrain = utils.nonDimensionalizeF(xTrain)
    NDTest = utils.nonDimensionalizeF(xTest)
    yTrainND = yTrain/NDTrain

    # Normalize dataset using mean&std
    xTrainN, xTrainM, xTrainStd = utils.normData(xTrain)
    xTestN = (xTest - xTrainM) / xTrainStd
    yTrainN, yTrainM, yTrainStd = utils.normData(yTrainND)

    # PCA SKLearn
    y = yTrainND - np.mean(yTrainND, axis=0) # Zero Mean
    pca = skd.PCA(n_components=0.99)
    dyn_pca = pca.fit_transform(y)
    yTrainN, yTrainM, yTrainStd = utils.normData(dyn_pca)
    #print("POD reduced output dimension to ",yTrainN.shape[1])

    import oldVerGPR 
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    gp2 = oldVerGPR.sklGPR(theta = [0.1, 1e-6, 10.0],regr = regr_type, corr = kernel_type,noiseLvl = 1e-14) # TODO: need to compare different trend functions and 
    gp2.fit(xTrainN, yTrainN)

    yPredictTraining2 = gp2.predict(xTrainN,eval_MSE=False, use_lib=True) * yTrainStd + yTrainM
    yPredictTesting2 = gp2.predict(xTestN,eval_MSE=False, use_lib=True) * yTrainStd + yTrainM
    # If trained using PCA data
    yPredictTraining2 = pca.inverse_transform(yPredictTraining2) + np.mean(yTrainND,axis=0)
    yPredictTesting2 = pca.inverse_transform(yPredictTesting2) + np.mean(yTrainND,axis=0)

    # Revert the Non-dimensionalization of forces
    yPredictTraining2 = yPredictTraining2 * NDTrain
    yPredictTesting2 = yPredictTesting2 * NDTest

    val_error = utils.errorCalc(yPredictTesting2,yTest, func=err_type)

    return val_error

def gpr_old_noND(xTrain, xTest, yTrain, yTest,regr_type,kernel_type,err_type):

    # Normalize dataset using mean&std
    xTrainN, xTrainM, xTrainStd = utils.normData(xTrain)
    xTestN = (xTest - xTrainM) / xTrainStd
    yTrainN, yTrainM, yTrainStd = utils.normData(yTrain)

    # PCA SKLearn
    y = yTrain - np.mean(yTrain, axis=0) # Zero Mean
    pca = skd.PCA(n_components=0.99)
    dyn_pca = pca.fit_transform(y)
    yTrainN, yTrainM, yTrainStd = utils.normData(dyn_pca)
    #print("POD reduced output dimension to ",yTrainN.shape[1])

    import oldVerGPR 
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    gp2 = oldVerGPR.sklGPR(theta = [0.1, 1e-6, 10.0],regr = regr_type, corr = kernel_type,noiseLvl = 1e-14) # TODO: need to compare different trend functions and 
    gp2.fit(xTrainN, yTrainN)

    yPredictTraining2 = gp2.predict(xTrainN,eval_MSE=False, use_lib=True) * yTrainStd + yTrainM
    yPredictTesting2 = gp2.predict(xTestN,eval_MSE=False, use_lib=True) * yTrainStd + yTrainM
    # If trained using PCA data
    yPredictTraining2 = pca.inverse_transform(yPredictTraining2) + np.mean(yTrain,axis=0)
    yPredictTesting2 = pca.inverse_transform(yPredictTesting2) + np.mean(yTrain,axis=0)

    val_error = utils.errorCalc(yPredictTesting2,yTest, func=err_type)

    return val_error


if __name__=="__main__":
    # initialize the dataset from files    
    dataset = dataset.Dataset(RAW_DATA_PATH, PARAM_PATH)
    # get k-folds
    folds = pickle.load(open("./folds.pckl","rb"))

    kernel_type = ['matern52','squared_exponential']
    err_type = ['rl2', 'rlinf']

    error_data_all = {}

    error_data = {}
    print("Running GPR New")
    for kernel in kernel_type:
        error_data[kernel] = {}
        for err in err_type:
            error_data[kernel][err] = []
            for i in range(len(folds.keys())):
                data = dataset.create_dataset(folds[i])
                x_train = data["train"]["x"]
                y_train = data["train"]["y"]
                x_test = data["test"]["x"]
                y_test = data["test"]["y"]

                print(f"\t{kernel} {err} {i}")
                val_error = gpr_new(x_train, x_test, y_train, y_test,kernel,err)
                error_data[kernel][err].append(val_error)
    error_data_all["new"] = error_data

    error_data = {}
    print("Running GPR Old - constant")
    for kernel in kernel_type:
        error_data[kernel] = {}
        for err in err_type:
            error_data[kernel][err] = []
            for i in range(len(folds.keys())):
                data = dataset.create_dataset(folds[i])
                x_train = data["train"]["x"]
                y_train = data["train"]["y"]
                x_test = data["test"]["x"]
                y_test = data["test"]["y"]

                print(f"\t{kernel} {err} {i}")
                val_error_c = gpr_old(x_train, x_test, y_train, y_test,'constant',kernel,err)
                error_data[kernel][err].append(val_error_c)
    error_data_all["old_constant"] = error_data

    error_data = {}
    print("Running GPR Old - linear")
    for kernel in kernel_type:
        error_data[kernel] = {}
        for err in err_type:
            error_data[kernel][err] = []
            for i in range(len(folds.keys())):
                data = dataset.create_dataset(folds[i])
                x_train = data["train"]["x"]
                y_train = data["train"]["y"]
                x_test = data["test"]["x"]
                y_test = data["test"]["y"]

                print(f"\t{kernel} {err} {i}")
                val_error_l = gpr_old(x_train, x_test, y_train, y_test,'linear',kernel,err)
                error_data[kernel][err].append(val_error_l)
    error_data_all["old_linear"] = error_data

    error_data = {}
    print("Running GPR Old without ND - constant")
    for kernel in kernel_type:
        error_data[kernel] = {}
        for err in err_type:
            error_data[kernel][err] = []
            for i in range(len(folds.keys())):
                data = dataset.create_dataset(folds[i])
                x_train = data["train"]["x"]
                y_train = data["train"]["y"]
                x_test = data["test"]["x"]
                y_test = data["test"]["y"]

                print(f"\t{kernel} {err} {i}")
                val_error_c = gpr_old_noND(x_train, x_test, y_train, y_test,'constant',kernel,err)
                error_data[kernel][err].append(val_error_c)
    error_data_all["old_noND_constant"] = error_data

    error_data = {}
    print("Running GPR Old without ND - linear")
    for kernel in kernel_type:
        error_data[kernel] = {}
        for err in err_type:
            error_data[kernel][err] = []
            for i in range(len(folds.keys())):
                data = dataset.create_dataset(folds[i])
                x_train = data["train"]["x"]
                y_train = data["train"]["y"]
                x_test = data["test"]["x"]
                y_test = data["test"]["y"]

                print(f"\t{kernel} {err} {i}")
                val_error_l = gpr_old_noND(x_train, x_test, y_train, y_test,'linear',kernel,err)
                error_data[kernel][err].append(val_error_l)
    error_data_all["old_noND_linear"] = error_data
    
    pickle.dump(error_data_all,open("error_data.pckl","wb"))