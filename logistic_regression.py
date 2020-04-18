import numpy as np


class LogisticRegression:
    def __init__(self, k):
        self.theta = np.random.normal(0, 1, size=(k+1,))
        self.mask = np.ones((k+1,))
        self.mask[0] = 0

    @staticmethod
    def sigmoid(x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    def prob_y_under_x(self, x, y):
        p1 = self.sigmoid(x, self.theta)
        return p1 if y == 1 else 1 - p1

    def fit(self, training_data, triggers, measure, alpha, beta, **kwargs):
        history = []
        for i, sample in enumerate(training_data):
            x, y = sample[0], sample[1]
            self.theta += alpha * ((y - self.sigmoid(x, self.theta)) * x - beta * self.theta * self.mask)
            if i + 1 in triggers:
                history.append(measure(self))
        return history

    def classify(self, x):
        return 1 if self.sigmoid(x, self.theta) >= 0.5 else 0
