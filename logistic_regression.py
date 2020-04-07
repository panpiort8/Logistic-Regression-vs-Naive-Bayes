import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data_pos = list()
    data_neg = list()
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


def sigmoid(x, teta):
    return 1/(1+np.exp(-np.dot(x, teta)))


def accuracy(data, teta):
    ok = 0
    for sample in data:
        x, y = sample[0], sample[1]
        h = sigmoid(x, teta)
        y0 = 1 if h >= 0.5 else 0
        if y0 == y:
            ok += 1
    return ok/len(data)


measures = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

data_pos, data_neg = load_data("rp.data")

alpha = 0.001
beta = 0
rounds = 10

histories = []
for i in range(rounds):
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    training_data, test_data = split_data(data_pos, data_neg)
    np.random.shuffle(training_data)
    where = [int(x * len(training_data)) for x in measures]

    teta = np.random.normal(scale=0.0001, size=(len(training_data[0][0]),))
    history = []
    for i, sample in enumerate(training_data):
        x, y = sample[0], sample[1]
        teta += alpha*((y-sigmoid(x, teta))*x - beta*teta)
        if i+1 in where:
            history.append(accuracy(training_data, teta))
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