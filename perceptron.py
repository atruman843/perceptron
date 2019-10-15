import random

class Perceptron:

    def __init__(self, features, labels, learnRate):
        self.epochs = 20
        self.features = features
        self.labels = labels
        self.learnRate = learnRate
        self.correct = 0
        # start with small, non-zero initial weights
        self.weights = [0.001] * len(self.features[0])
        self.delta = [0] * len(self.features[0])

    def train(self):

        total = len(self.features)

        # iterate through for 20 epochs
        for i in range(0, self.epochs):

            self.delta = [0] * len(self.features[0])
            self.correct = 0
            test_set = [0] * 10

            # create the new test_set with a random set of indices to test
            for j in range(0, len(test_set)):
                ind = random.randint(0, 99)
                while ind in test_set:
                    ind = random.randint(0, 99)
                test_set[j] = ind

            # iterate through all of the features
            for index1, feature in enumerate(self.features):

                prediction = self.predict(feature)

                # collect the incorrect predictions and perform loss on these features
                if not(index1 in test_set) and not(prediction == self.labels[index1]):
                    self.lossCalculation(prediction, index1)

            # update the weights with the accumulated loss
            for index in range(0, len(self.weights)):
                self.weights[index] += self.delta[index]

            # test how the updated perceptron performs on our 10 test samples
            for index in test_set:
                prediction = self.predict(self.features[index])
                if prediction == self.labels[index]:
                    self.correct += 1

            print '{percentCorrect}%'.format(percentCorrect = (float(self.correct)/len(test_set))*100, i = i+1)

    def lossCalculation(self, prediction, index1):
        for index, weight in enumerate(self.weights):
            self.delta[index] += self.learnRate * float(self.labels[index1] - prediction) * self.features[index1][index]

    def predict(self, feature):

        prediction = 0
        for index, val in enumerate(feature):
            k = val * self.weights[index]
            prediction += k
        confidence = abs(prediction)
        if prediction > 0:
            prediction = 1
        else:
            prediction = -1
        print 'Confidence: {conf} for prediction: {pred}'.format(conf=confidence, pred=prediction)
        return prediction
