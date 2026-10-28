import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=4601):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Number of samples and features
        n_samples, n_features = X.shape #see the part "X and Y"

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training
        for _ in range(self.n_iterations):

            for i in range(n_samples):

                # Linear combination
                z = np.dot(X[i], self.weights) + self.bias

                # Prediction
                y_pred = 1 if z >= 0 else 0

                # Update only if prediction is wrong
                if y_pred != y[i]:
                    self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                    self.bias += self.learning_rate * (y[i] - y_pred)

        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.where(z >= 0, 1, 0)