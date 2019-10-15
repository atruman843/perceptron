import argparse
from file_reader import FileReader
from perceptron import Perceptron

parser = argparse.ArgumentParser(description = 'This script processes data in the \
                                 data folder through the perceptron.')

parser.add_argument('-n', '--learnRate', action='store', help='Specify the learning rate of the perceptron.', required=True)

args = parser.parse_args()

n = float(args.learnRate)

reader = FileReader()
reader.read()
features = reader.get_features()
labels = reader.get_labels()

perceptron = Perceptron(features, labels, n)
perceptron.train()
