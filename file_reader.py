from scipy.io import arff
from perceptron import Perceptron

class FileReader:

    def __init__(self):
        self.features = []
        self.labels = []

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def read(self):
        data, meta = arff.loadarff('data/iris.arff')
        for row in data:
            self.features.append([1, row[0], row[1], row[2], row[3]])
            if (row[4] == 'Iris-setosa'):
                self.labels.append(1)
            else:
                self.labels.append(-1)
