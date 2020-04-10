import numpy as np
import matplotlib.pyplot as plt
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression


def load_data(path):
    data_pos, data_neg = [], []
    with open(path, 'r') as file:
        for line in file:
            line = line.split()
            X = np.array([float(x) for x in line[:-1]])
            y = int(line[-1]) // 4
            if y == 1:
                data_pos.append((X, y))
            else:
                data_neg.append((X, y))
    return data_pos, data_neg


def split_data(data_pos, data_neg, p=2 / 3):
    len_pos = int(len(data_pos) * p)
    training_pos = data_pos[:len_pos]
    test_pos = data_pos[len_pos:]

    len_neg = int(len(data_neg) * p)
    training_neg = data_neg[:len_neg]
    test_neg = data_neg[len_neg:]
    return training_neg + training_pos, test_pos + test_neg


def accuracy(model, data):
    ok = 0
    for sample in data:
        x, y = sample[0], sample[1]
        y0 = model.classify(x)
        if y0 == y:
            ok += 1
    return ok / len(data)


def loss(model, data):
    loss = 0
    for x, y in data:
        loss += (1 - model.prob_y_under_x(x, y)) ** 2
    return loss / len(data)


def feed_with_data(measure, data):
    def func(model):
        return measure(model, data)

    return func


def test(cls, partial_triggers, data_pos, data_neg, rounds, measure, alpha=None, beta=None):
    histories = []
    for i in range(rounds):
        np.random.shuffle(data_pos)
        np.random.shuffle(data_neg)
        training_data, test_data = split_data(data_pos, data_neg)
        np.random.shuffle(training_data)
        triggers = [int(x * len(training_data)) for x in partial_triggers]
        model = cls(k=9)
        history = model.fit(training_data, triggers, feed_with_data(measure, test_data), alpha=alpha, beta=beta)
        histories.append(history)

    return np.mean(histories, axis=0)


partial_triggers = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
data_pos, data_neg = load_data("rp.data")

rounds = 10
logistic_history = test(LogisticRegression, partial_triggers, data_pos, data_neg, rounds, loss, alpha=0.005, beta=0.001)
bayes_history = test(NaiveBayes, partial_triggers, data_pos, data_neg, rounds, loss)

plt.figure()
plt.plot(partial_triggers, logistic_history, 'ro-', label="logistic_loss")
plt.plot(partial_triggers, bayes_history, 'bo-', label="bayes_loss")
plt.title("Naive Bayes vs Logistic Regression")
plt.xlabel("Part of training set")
plt.ylabel("Loss")
plt.legend()
plt.show()
