import numpy as np
from random import randrange

class smo_algo:
    def __init__(self, train_data_features, labels, reg_strength, tolerance):
        self.X = train_data_features
        self.c_labels = labels
        self.C = reg_strength
        self.tol = tolerance
        self.num_lagrange, self.theta_hat_dim = np.shape(self.X)
        self.lagrange_muls = np.zeros(self.num_lagrange)
        self.errors = np.zeros(self.num_lagrange)
        self.epsilon = 10**(-3)
        self.theta0_hat = 0
        self.theta_hat = np.zeros(self.theta_hat_dim)

    def discriminant_score(self, i):
        return self.theta0_hat + float(np.dot(self.theta_hat.T, self.X[i]))

    def compute_error(self, i, c):
        return self.discriminant_score(i) - c

    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.lagrange_muls > 0, self.lagrange_muls < self.C))[0]

    def get_maximum_change_lagrange(self, non_bound_indices, E2):
        i1 = -1
        max_step = 0
        for j in non_bound_indices:
            E1 = self.errors[j] - self.c_labels[j]
            step = abs(E1 - E2)
            if step > max_step:
                max_step = step
                i1 = j
        return i1

    def take_cordinate_ascent_step(self, i1, i2, c2, lambda2, X2, E2):
        if i1 == i2:
            return False

        lambda1 = self.lagrange_muls[i1]
        c1 = self.c_labels[i1]
        X1 = self.X[i1]
        E1 = self.compute_error(i1, c1)
        labels_prod = c1 * c2

        if c1 != c2:
            L = max(0, lambda2 - lambda1)
            H = min(self.C, lambda2 + lambda1)
        else:
            L = max(0, lambda2 + lambda1 - self.C)
            H = min(self.C, lambda2 + lambda1)

        if L == H:
            return False

        k11 = np.dot(X1.T, X1)
        k12 = np.dot(X1, X2)
        k22 = np.dot(X2, X2)
        second_derivative = k11 + k22 - 2 * k12

        if second_derivative <= 0:
            return False

        lambda2_new = lambda2 + c2 * (E1 - E2) / second_derivative
        lambda2_new = np.clip(lambda2_new, L, H)

        if abs(lambda2_new - lambda2) < self.epsilon:
            return False

        lambda1_new = lambda1 + labels_prod * (lambda2 - lambda2_new)

        theta0_hat1 = E1 + c1 * (lambda1_new - lambda1) * k11 + c2 * (lambda2_new - lambda2) * k12 + self.theta0_hat
        theta0_hat2 = E2 + c1 * (lambda1_new - lambda1) * k12 + c2 * (lambda2_new - lambda2) * k22 + self.theta0_hat

        if 0 < lambda1_new < self.C:
            self.theta0_hat = theta0_hat1
        elif 0 < lambda2_new < self.C:
            self.theta0_hat = theta0_hat2
        else:
            self.theta0_hat = (theta0_hat1 + theta0_hat2) / 2.0

        self.theta_hat += c1 * (lambda1_new - lambda1) * X1 + c2 * (lambda2_new - lambda2) * X2

        delta_theta0_hat = self.theta0_hat - ((theta0_hat1 + theta0_hat2) / 2.0)

        for i in range(self.num_lagrange):
            if 0 < self.lagrange_muls[i] < self.C:
                self.errors[i] += c1 * (lambda1_new - lambda1) * np.dot(X1, self.X[i]) + \
                                   c2 * (lambda2_new - lambda2) * np.dot(X2, self.X[i]) - delta_theta0_hat

        self.errors[i1] = 0
        self.errors[i2] = 0
        self.lagrange_muls[i1] = lambda1_new
        self.lagrange_muls[i2] = lambda2_new

        return True

    def check_example(self, i2):
        c2 = self.c_labels[i2]
        lambda2 = self.lagrange_muls[i2]
        X2 = self.X[i2]
        E2 = self.compute_error(i2, c2)
        constraint_diff2 = E2 * c2

        if not ((constraint_diff2 < -self.tol and lambda2 < self.C) or (constraint_diff2 > self.tol and lambda2 > 0)):
            return 0

        non_bound_indices = self.get_non_bound_indexes()
        i1 = self.get_maximum_change_lagrange(non_bound_indices, E2)

        if i1 >= 0 and self.take_cordinate_ascent_step(i1, i2, c2, lambda2, X2, E2):
            return 1

        for i1 in np.random.permutation(non_bound_indices):
            if self.take_cordinate_ascent_step(i1, i2, c2, lambda2, X2, E2):
                return 1

        for i1 in np.random.permutation(self.num_lagrange):
            if self.take_cordinate_ascent_step(i1, i2, c2, lambda2, X2, E2):
                return 1

        return 0

    def smo_algo_main_loop(self):
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.num_lagrange):
                    num_changed += self.check_example(i)
            else:
                for i in self.get_non_bound_indexes():
                    num_changed += self.check_example(i)

            examine_all = not examine_all if num_changed == 0 else examine_all

    def predict(self, X):
        return np.array([1 if self.theta0_hat + np.dot(self.theta_hat, x) >= 0 else -1 for x in X])
