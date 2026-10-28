import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    # Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Training
    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):

            # Linear combination
            z = np.dot(X, self.weights) + self.bias

            # Prediction as probability
            y_pred = self.sigmoid(z)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    # Predict probabilities
    def predict_proba(self, X):

        z = np.dot(X, self.weights) + self.bias

        return self.sigmoid(z)

    # Predict classes
    def predict(self, X):

        probabilities = self.predict_proba(X)

        return np.where(probabilities > 0.5, 1, 0)