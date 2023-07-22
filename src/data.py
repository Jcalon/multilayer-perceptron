import csv
import numpy as np
import pandas as pd
import random

class Datasets:
    """
    Create a dataset from a csv file
    Input : "./path/to/my/dataset.csv", "predict_column", [ "feature1", "feature2", ... ]
    """
    def __init__(self, filepath, predict_column, features_columns):
        self.data = self.read_csv(filepath)
        self.predict = self.data[predict_column]
        self.size = len(self.predict)
        self.classes = sorted(set(self.data[predict_column]))
        self.nb_classes = len(self.classes)
        self.features = self.filter(features_columns)
        self.nb_features = len(self.features)
        self.minmax = []
        self.scaled = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.train_size = None
        self.test_size = None

    def read_csv(self, filepath):
        """
        Create a json from a csv filepath
        Input : "./path/to/my/dataset.csv"
        Output: { 'a': [ "a1", "a2", ... ], 'b': [ "b1", "b2", ... ] , ... }
        """
        data = None
        self.size = 0
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if data is None:
                    labels = row
                    data = {label: [] for label in labels}
                else:
                    self.size += 1
                    for value, label in zip(row, labels):
                        data[label].append(value)
        return data

    def filter(self, keys):
        """
        select only some json key
        Convert their values to float or NaN if not exist
        Replace NaN by mean
        Input : [ "a", "z" ]
        Output : {'a' : [ (float) a1, (float) a2 , ... ], 'z': [ (float) z1, (float) z2 , ... ]}
        """
        ret = []
        for label in self.data:
            if label in keys:
                ret.append([float(line) if line else np.NAN for line in self.data[label]])
        for i in range(len(keys)):
            for j in range(self.size):
                if np.isnan(ret[i][j]):
                    data = [ret[i][k] for k in range(self.size) if self.predict[j] == self.predict[k] and not np.isnan(ret[i][k])]
                    ret[i][j] = sum(data) / len(data)
        return ret

    def minmaxScaler(self, data, minmax=None):
        """
        Scale data between 0 and 1
        Input : [ 800, 0, -800, ... ]
        Output : [ 1, 0.5, 0, ... ]
        """
        dmin = min(data) if minmax is None or min(data) < minmax['min'] else minmax['min']
        dmax = max(data) if minmax is None or max(data) > minmax['max'] else minmax['max']
        self.minmax.append({'min': dmin, 'max': dmax})
        return [((x - dmin) / (dmax - dmin)) for x in data]

    def scale(self, minmax=None):
        """
        Scale all features
        Input : [ [ 800,0, -800, ... ] , ...]
        Output : [ [ 1, 0.5, 0, ... ] , ... ]
        """
        self.scaled = []
        for i in range(self.nb_features):
            self.scaled.append(self.minmaxScaler(self.features[i], None if minmax is None else minmax[i]))
        self.X = self.transform(self.scaled)
        self.Y = self.get_Y(self.predict)
        return self.X, self.Y

    def transform(self, features):
        """
        Transform a list of features to a np.array
        Input : [ [ a1, a2, a3, ... ], [ b1, b2, b3, ... ], [ c1, c2, c3, ... ], ... ]
        Output : (np.array) [ [ a1, b1, c1, ... ], [ a2, b2, c2, ... ], [ a2, b2, c2, ... ], ... ]
        """
        n = len(features)
        size = len(features[0])
        ret = np.ndarray((size, n), float)
        for i in range(n):
            for j in range(size):
                ret[j, i] = float(features[i][j])
        return ret

    def get_Y(self, to_predict):
        """
        Transform a list of classes to a np.array
        Input : [ "class1", "class2", ... ]
        Output : (np.array) [ 1, 2, ... ]
        """
        Y = []
        for row in to_predict:
            Y.append(self.classes.index(row))
        return np.array(Y)

    def shuffle(self, X, Y):
        """
        Shuffle X and Y
        Input : [ [ a1, a2, a3, ... ], [ b1, b2, b3, ... ], [ c1, c2, c3, ... ], ... ]  , [0, 1, 2, ...]
        Output : [ [ a3, a1, a2, ... ], [ b3, b1, b2, ... ], [ c3, c1, c2, ... ], ... ]  , [2, 0, 1, ...]
        """
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

    def train_test_split(self, percent_train=80):
        """
        Split X and Y into X_train, Y_train, X_test and Y_test
        """
        if percent_train > 100 or percent_train < 0:
            raise ValueError("percent_train must be between 0 and 100")
        self.shuffle(self.X, self.Y)
        self.train_size = int(self.size * percent_train / 100)
        self.test_size = self.size - self.train_size
        self.Y_test = np.array(self.Y[:self.test_size])
        self.Y_train = np.array(self.Y[self.test_size:])
        self.X_test = np.array(self.X[:self.test_size])
        self.X_train = np.array(self.X[self.test_size:])
