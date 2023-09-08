"""
Neural Network with 1 hidden layer from scratch
By: Sebastian Mora (@Bastian1110)
"""

# These libraries provide essential functionalities for data manipulation,
# mathematical operations, and plotting.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Read data from a CSV file into a DataFrame.
df = pd.read_csv("WineQT.csv")

# Drop the "Id" column as it's not necessary for modeling purposes.
df.drop("Id", axis=1, inplace=True)

# Separate the dataset into features (X) and the target variable (y).
X = df.drop("quality", axis=1).values
y = df["quality"].values

# Normalize the features to ensure they have a mean of 0 and standard deviation of 1.
# This makes the training process smoother and more stable.
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the dataset into training, validation, and test sets. This ensures that we can
# train the model, tune its parameters, and then evaluate its performance on unseen data.
def rearrange(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # Reshape the target arrays to be column vectors. This ensures compatibility during matrix operations.
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, X_test, X_val, y_train, y_test, y_val


all_test_losses = []
all_accuracies = []


for i in range(10):
    print(f"Run number : {i}")

    X_train, X_test, X_val, y_train, y_test, y_val = rearrange(X, y)

    # Define the neural network's architecture: Input layer size, hidden layer size, and output layer size.
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = 1

    # Initialize weights and biases for the neural network. Weights are initialized with random values,
    # and biases are initialized with zeros.
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))

    # Define hyperparameters for training.
    learning_rate = 0.01
    epochs = 10000

    # Lists to store loss values during training for visualization.
    losses_train = []
    losses_val = []

    for epoch in range(epochs):
        # FORWARD PROPAGATION:
        # 1. Calculate the weighted sum for the hidden layer and apply the tanh activation function.
        # 2. Calculate the weighted sum for the output layer.
        # (Note: No activation function is applied in the output layer in this regression task)
        weighted_sum_hidden = np.dot(X_train, weights_input_hidden) + bias_hidden
        activation_hidden = np.tanh(weighted_sum_hidden)
        weighted_sum_output = (
            np.dot(activation_hidden, weights_hidden_output) + bias_output
        )
        activation_output = weighted_sum_output

        # Compute the mean squared error loss for the training data.
        loss = mean_squared_error(y_train, activation_output)
        losses_train.append(loss)

        # Calculate the mean squared error loss for the validation data.
        # This provides insight into how well the model is generalizing.
        weighted_sum_hidden_val = np.dot(X_val, weights_input_hidden) + bias_hidden
        activation_hidden_val = np.tanh(weighted_sum_hidden_val)
        weighted_sum_output_val = (
            np.dot(activation_hidden_val, weights_hidden_output) + bias_output
        )
        activation_output_val = weighted_sum_output_val
        loss_val = mean_squared_error(y_val, activation_output_val)
        losses_val.append(loss_val)

        # Print training progress every 100 epochs.
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Training Loss: {loss}, Validation Loss: {loss_val}")

        # BACKPROPAGATION:
        # Compute the gradients for the weights and biases.
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

        # Update the weights and biases using the computed gradients and learning rate.
        weights_input_hidden -= learning_rate * gradient_weights_input_hidden
        bias_hidden -= learning_rate * gradient_bias_hidden
        weights_hidden_output -= learning_rate * gradient_weights_hidden_output
        bias_output -= learning_rate * gradient_bias_output

    # Use the trained neural network to make predictions on the test set.
    weighted_sum_hidden_test = np.dot(X_test, weights_input_hidden) + bias_hidden
    activation_hidden_test = np.tanh(weighted_sum_hidden_test)
    weighted_sum_output_test = (
        np.dot(activation_hidden_test, weights_hidden_output) + bias_output
    )
    activation_output_test = weighted_sum_output_test

    # Calculate the mean squared error loss for the test data.
    test_loss = mean_squared_error(y_test, activation_output_test)
    print(f"Evaluation Loss: {test_loss}")

    # Function to compute accuracy based on a given threshold.
    def accuracy(y_real, y_pred, threshold=0.5):
        """Compute accuracy considering predictions within a certain threshold as correct."""
        correct = np.abs(y_real - y_pred) <= threshold
        accuracy = np.mean(correct)
        return accuracy

    # Compute the accuracy for the test data.
    acc = accuracy(y_test, activation_output_test)
    print(f"Accuracy: {acc * 100:.2f}%")

    all_test_losses.append(test_loss)
    all_accuracies.append(acc)

    # Convert continuous quality values into binary classification (good or not good).
    y_test_classified = (y_test > 6).astype(int)
    predictions_classified = (activation_output_test > 6).astype(int)

    # Display confusion matrix and classification report for the binary classification results.
    print(confusion_matrix(y_test_classified, predictions_classified))
    print(classification_report(y_test_classified, predictions_classified))


# Plot the test losses and accuracies after all runs
plt.figure(figsize=(10, 5))

# Plotting test losses
plt.subplot(1, 2, 1)
plt.plot(all_test_losses, marker="o", linestyle="--", color="r")
plt.title("Test Losses over 10 runs")
plt.xlabel("Run")
plt.ylabel("Loss")

# Plotting accuracies
plt.subplot(1, 2, 2)
plt.plot(all_accuracies, marker="o", linestyle="--", color="b")
plt.title("Accuracies over 10 runs")
plt.xlabel("Run")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
