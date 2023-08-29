"""
    Neural Network with 1 hidden layer from scatch
    By : Sebastian Mora (@Bastian1110)
"""
# Load data from csv and drop quality values
import pandas as pd

df = pd.read_csv("WineQT.csv")
df.drop("Id", axis=1, inplace=True)

X = df.drop("quality", axis=1).values
y = df["quality"].values

# Fucntion to plot the NN (credit to ChatGPT)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_nn_with_weights(input_size, hidden_size, output_size, W1, W2, epoch):
    plt.clf()
    G = nx.DiGraph()
    pos = {}
    layers = [input_size, hidden_size, output_size]
    count = 0
    for i, layer in enumerate(layers):
        for j in range(layer):
            G.add_node(count)
            pos[count] = (i, j - layer / 2)
            count += 1
    start = 0
    edge_weights = {}
    for i, layer in enumerate(layers[:-1]):
        weights = W1 if i == 0 else W2
        for j in range(start, start + layer):
            for k, weight in enumerate(weights[j - start]):
                target = start + layer + k
                G.add_edge(j, target)
                edge_weights[(j, target)] = np.abs(weight)
        start += layer

    edges = G.edges()
    weights = [edge_weights[edge] for edge in edges]
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="salmon",
        font_weight="italic",
        node_size=700,
        font_size=14,
        width=weights,
        edge_color="lightblue",
    )
    plt.title(f"Neural Network in epoch :  {epoch}")
    plt.pause(0.1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Scale the data from the dataset to values from -2 to 2
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divide the dataset into train and test groups
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Transforming the y values to a matrix of 1 column
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

from sklearn.metrics import mean_squared_error

# Starting the Neural Network
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Parameters for training
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    weighted_sum_hidden = np.dot(X_train, weights_input_hidden) + bias_hidden
    activation_hidden = np.tanh(weighted_sum_hidden)
    weighted_sum_output = np.dot(activation_hidden, weights_hidden_output) + bias_output
    activation_output = weighted_sum_output

    # Loss calculation
    loss = mean_squared_error(y_train, activation_output)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
        plot_nn_with_weights(
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            epoch,
        )

    # Backpropagation
    gradient_weighted_sum_output = activation_output - y_train
    gradient_weights_hidden_output = np.dot(
        activation_hidden.T, gradient_weighted_sum_output
    ) / len(y_train)
    gradient_bias_output = np.sum(
        gradient_weighted_sum_output, axis=0, keepdims=True
    ) / len(y_train)

    gradient_weighted_sum_hidden = np.dot(
        gradient_weighted_sum_output, weights_hidden_output.T
    ) * (1 - np.power(activation_hidden, 2))
    gradient_weights_input_hidden = np.dot(
        X_train.T, gradient_weighted_sum_hidden
    ) / len(y_train)
    gradient_bias_hidden = np.sum(
        gradient_weighted_sum_hidden, axis=0, keepdims=True
    ) / len(y_train)

    # Updating the weights
    weights_input_hidden -= (
        learning_rate * gradient_weights_input_hidden
    )  # Corrected this line
    bias_hidden -= learning_rate * gradient_bias_hidden
    weights_hidden_output -= learning_rate * gradient_weights_hidden_output
    bias_output -= learning_rate * gradient_bias_output

# Evaluation with x and y test
weighted_sum_hidden_test = np.dot(X_test, weights_input_hidden) + bias_hidden
activation_hidden_test = np.tanh(weighted_sum_hidden_test)
weighted_sum_output_test = (
    np.dot(activation_hidden_test, weights_hidden_output) + bias_output
)
activation_output_test = weighted_sum_output_test

test_loss = mean_squared_error(y_test, activation_output_test)
print(f"Evaluation Loss: {test_loss}")


def accuracy(y_real, y_pred, umbral=0.5):
    correct = np.abs(y_real - y_pred) <= umbral
    accuracy = np.mean(correct)
    return accuracy


# Suponiendo que A2_test son tus predicciones y y_test son las etiquetas reales
accuracy = accuracy(y_test, activation_output_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

from sklearn.neural_network import MLPRegressor

# Creating a MLP Regressor (NN with Adam)
model = MLPRegressor(
    hidden_layer_sizes=(10,), activation="relu", max_iter=1000000, random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate Mean Squared Error
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Train Error: {mse_train}, Test Error: {mse_test}")

# Get acurracy
accuracy_train = accuracy(y_train, y_pred_train)
accuracy_test = accuracy(y_test, y_pred_test)
print(f"Train Accuracy: {accuracy_train}%, Test Accuracy: {accuracy_test}%")
