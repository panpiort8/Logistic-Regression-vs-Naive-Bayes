class NaiveBayes:
    def __init__(self, k):
        self.k = k
        self.stats = [[[dict() for i in range(k)], 0], [[dict() for i in range(k)], 0]]

    # returns p(x_j=a|y)
    def prob_x_j_under_y(self, j, a, y):
        count = self.stats[y][0][j].get(a, 0)
        return (1 + count) / (2 + self.stats[y][1])

    def prob_y(self, y):
        return (1 + self.stats[y][1]) / (2 + self.stats[0][1] + self.stats[1][1])

    def prob_x_and_y(self, x, y):
        prob = self.prob_y(y)
        for j, a in enumerate(x):
            prob *= self.prob_x_j_under_y(j, a, y)
        return prob

    def prob_y_under_x(self, x, y):
        x_and_0 = self.prob_x_and_y(x, 0)
        x_and_1 = self.prob_x_and_y(x, 1)
        p0 = x_and_0 / (x_and_0 + x_and_1)
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
