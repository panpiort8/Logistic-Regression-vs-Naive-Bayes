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
    def __init__(self, theta_size):
        self.theta = np.random.normal(scale=0.0001, size=theta_size)

    @staticmethod
    def sigmoid(x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    def fit(self, training_data, func_triggers, measure, alpha, beta):
        history = []
        for i, sample in enumerate(training_data):
            x, y = sample[0], sample[1]
            self.theta += alpha * ((y - self.sigmoid(x, self.theta)) * x - beta * self.theta)
            if i + 1 in func_triggers:
                history.append(measure(self.classify))
        return history

    def classify(self, x):
        return 1 if self.sigmoid(x, self.theta) >= 0.5 else 0


def accuracy(classify, data):
    ok = 0
    for sample in data:
        x, y = sample[0], sample[1]
        y0 = classify(x)
        if y0 == y:
            ok += 1
    return ok/len(data)


def feed_with_data(measure, data):
    def func(classify):
        return measure(classify, data)
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

    model = LogisticRegression(theta_size=(9,))
    history = model.fit(training_data, triggers, feed_with_data(accuracy, test_data), alpha, beta)
    histories.append(history)

history = np.mean(histories, axis=0)

plt.figure()
plt.plot(measures, history, 'ro', label="train_loss")
plt.xscale("log")
plt.title("Training Accuracy")
plt.xlabel("Part of training set")
plt.ylabel("Accuracy")
plt.legend()
plt.show()