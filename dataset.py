import csv
import numpy as np
import pickle

# create X and Y datasets from the raw data files and the parameter file
class Dataset:

    def __init__(self, raw_data_path, param_path):
        self._raw_path = raw_data_path
        
        self._parameters = []
        with open(param_path) as f:
            csv_file = csv.reader(f)
            next(csv_file) # skip column names
            for r in csv_file:
                self._parameters.append([int(r[0])]+[float(i) for i in r[1:11]])

    def create_dataset(self, split_data):

        train_idxs = split_data["train"]
        test_idxs = split_data["test"]

        dataset = {}
        dataset["train"] = self._get_data(train_idxs)
        dataset["test"] = self._get_data(test_idxs)

        return dataset


    def _get_data(self, case_idxs):

        dataset = {"x":np.empty((len(case_idxs),3)), #len(self._parameters[0])-1)),
                    "y": np.zeros((len(case_idxs),6001))}

        for i,idx in enumerate(case_idxs):
            
            # open the data file
            path = f'{self._raw_path}/RUN_{idx}'
            filename = path + '/2D_FZ_FORMATTED.txt'
            _,y1,y2 = pickle.load(open(filename, 'rb'))  # x values, y1 values, y2 values

            #data stored as single element tuples which makes it hard to see the 2d shape 
            # so converting it to standard 1d array 
            # some of them have different number of points - ask kalki about this
            dataset["y"][i,0:len(y1)] = y1[:]

            # store param data
            dataset["x"][i,:] = self._parameters[idx][3:6] # only using aoa = 0

        return dataset

