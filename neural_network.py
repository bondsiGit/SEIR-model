import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

class NeuralNetworkModel:
    def __init__(self, hidden_layer_sizes=(64, 64), max_iter=500, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def train(self, X, y):
        """
        Train the neural network model using scikit-learn's MLPRegressor.
        :param X: Input features (e.g., time).
        :param y: Target values (e.g., residuals).
        :return: Trained model and scaler.
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # Scale the data
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build the neural network
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        return self.model, self.scaler

    def predict(self, X):
        """
        Predict using the trained neural network model.
        :param X: Input features.
        :return: Predicted values.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)