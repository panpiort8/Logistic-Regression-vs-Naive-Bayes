import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data_pos, data_neg = [], []
    with open(path, 'r') as file:
        for line in file:
            line = line.split()
            X = np.array([float(x) for x in line[:-1]])
            y = int(line[-1]) // 4
            if y==1:
                data_pos.append((X, y))
            else:
                data_neg.append((X, y))
    return data_pos, data_neg


def split_data(data_pos, data_neg, p=2/3):
    len_pos = int(len(data_pos) * p)
    training_pos = data_pos[:len_pos]
    test_pos = data_pos[len_pos:]

    len_neg = int(len(data_neg) * p)
    training_neg = data_neg[:len_neg]
    test_neg = data_neg[len_neg:]
    return training_neg + training_pos, test_pos+test_neg


class LogisticRegression:
    def __init__(self, k):
        self.theta = np.random.normal(scale=0.0001, size=(k,))

    @staticmethod
    def sigmoid(x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    def prob_y_under_x(self, x, y):
        p1 = self.sigmoid(x, self.theta)
        return p1 if y == 1 else 1-p1

    def fit(self, training_data, triggers, measure, alpha, beta, **kwargs):
        history = []
        for i, sample in enumerate(training_data):
            x, y = sample[0], sample[1]
            self.theta += alpha * ((y - self.sigmoid(x, self.theta)) * x - beta * self.theta)
            if i + 1 in triggers:
                history.append(measure(self))
        return history

    def classify(self, x):
        return 1 if self.sigmoid(x, self.theta) >= 0.5 else 0


class NaiveBayes:
    def __init__(self, k):
        self.k = k
        self.stats = [[[dict() for i in range(k)], 0], [[dict() for i in range(k)], 0]]

    # returns p(x_j=a|y)
    def prob_x_j_under_y(self, j, a, y):
        count = self.stats[y][0][j].get(a, 0)
        return (1+count)/(2+self.stats[y][1])

    def prob_y(self, y):
        return (1+self.stats[y][1])/(2+self.stats[0][1]+self.stats[1][1])

    def prob_x_and_y(self, x, y):
        prob = self.prob_y(y)
        for j, a in enumerate(x):
            prob *= self.prob_x_j_under_y(j, a, y)
        return prob

    def prob_y_under_x(self, x, y):
        x_and_0 = self.prob_x_and_y(x, 0)
        x_and_1 = self.prob_x_and_y(x, 1)
        p0 = x_and_0 / (x_and_0+x_and_1)
        return p0 if y == 0 else 1 - p0

    def classify(self, x):
        p0 = self.prob_y_under_x(x, 0)
        return 0 if p0 >= 0.5 else 1

    def fit(self, training_data, triggers, measure, **kwargs):
        history = []
        for i, sample in enumerate(training_data):
            x, y = sample[0], sample[1]
            for j, a in enumerate(x):
                dictionary = self.stats[y][0][j]
                dictionary[a] = dictionary.setdefault(a, 0) + 1
            self.stats[y][1] += 1
            if i + 1 in triggers:
                history.append(measure(self))
        return history


def accuracy(model, data):
    ok = 0
    for sample in data:
        x, y = sample[0], sample[1]
        y0 = model.classify(x)
        if y0 == y:
            ok += 1
    return ok/len(data)


def loss(model, data):
    loss = 0
    for x, y in data:
        loss += (1-model.prob_y_under_x(x, y))**2
    return loss / len(data)


def feed_with_data(measure, data):
    def func(model):
        return measure(model, data)
    return func


measures = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

data_pos, data_neg = load_data("rp.data")

alpha = 0.001
beta = 0.001
rounds = 10

histories = []
for i in range(rounds):
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    training_data, test_data = split_data(data_pos, data_neg)
    np.random.shuffle(training_data)
    triggers = [int(x * len(training_data)) for x in measures]

    model = NaiveBayes(k=9)
    history = model.fit(training_data, triggers, feed_with_data(loss, test_data))
    # model = LogisticRegression(k=9)
    # history = model.fit(training_data, triggers, feed_with_data(accuracy, test_data), alpha, beta)
    histories.append(history)

history = np.mean(histories, axis=0)

plt.figure()
plt.plot(measures, history, 'ro', label="test_loss")
plt.xscale("log")
plt.title("Training Accuracy")
plt.xlabel("Part of training set")
plt.ylabel("Accuracy")
plt.legend()
plt.show()