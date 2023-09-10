"""
Neural Network to classify Wine Quality Dataset
By: Sebastian Mora (@Bastian1110)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from provided URL
url = "WineQT.csv"
data = pd.read_csv(url)


# Drop the unnecessary Id column if present
data = data.drop(
    columns=["Id"], errors="ignore"
)  # Added errors="ignore" to handle case where 'Id' is not present


# Convert quality values into one of three categories for multi-class classification:
# bad (0), regular (1), and good (2)
def classify_quality(quality):
    if quality <= 4:
        return 0  # bad
    elif 5 <= quality <= 6:
        return 1  # regular
    else:
        return 2  # good


data["quality"] = data["quality"].apply(classify_quality)

# Separate the data into features (X) and target label (y)
X = data.drop(columns=["quality"])
y = data["quality"]

# Split the data into training, validation, and test sets
# First, split data into a temporary set and test set (80% and 20% of total data respectively)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Then, split the temporary set into actual training and validation sets (60% and 20% of total data respectively)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25
)  # 0.25 x 0.8 = 0.2

# Standardize the feature values using StandardScaler to have a mean of 0 and variance of 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert the class labels to one-hot encoded vectors for neural network training
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Model training and evaluation

epochs = 250
num_trials = 10
history_list = []  # To store training history of each trial
metrics_list = []  # To store evaluation metrics of each trial

for trial in range(num_trials):
    print(f"\nTraining Trial {trial + 1}/{num_trials}")

    # Neural Network Architecture Parameter and Configuration:
    # Input Layer: The size corresponds to the number of features in the dataset.
    #
    # Hidden Layers:
    # First Layer 128 Neurons with ReLU Activation : A larger layer can capture complex patterns in the data. ReLU is chosen for its efficiency and ability to deal with the vanishing gradient problem.
    # Second Layer 64 Neurons with ReLU Activation : This layer can refine patterns captured by the previous layer. The size is halved to gradually consolidate the information.
    # Third Layer 32 Neurons with ReLU Activation : Further refines the learned representations
    #
    # Output Layer 3 Neurons with Softmax Activation):Corresponds to the three wine quality categories (bad, regular, good). Softmax ensures the output values are probabilities that sum to 1, making it easier to pick the most likely category.
    #
    # Optimizer 'Adam' Adjusts learning rates throughout training, ensuring a good balance between speed and accuracy.
    #
    # Loss 'Categorical Crossentropy' As this is a multi-class classification task, this loss function is apt as it penalizes incorrect classifications

    # Build a neural network model
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(3, activation="softmax"))  # 3 output units for 3 classes
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model using the training data
    history = model.fit(
        X_train,
        y_train_onehot,
        validation_data=(X_val, to_categorical(y_val)),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )
    history_list.append(history)

    # Evaluate the trained model on test data and store the metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    metrics = classification_report(
        y_test,
        y_pred_classes,
        target_names=["bad", "regular", "good"],
        output_dict=True,
    )
    metrics_list.append(metrics)

    # Print the classification report and confusion matrix for each trial
    print(metrics)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

#  Visualization of results

# Calculate average metrics over the 10 trials
precision_list = [trial["weighted avg"]["precision"] for trial in metrics_list]
recall_list = [trial["weighted avg"]["recall"] for trial in metrics_list]
f1_list = [trial["weighted avg"]["f1-score"] for trial in metrics_list]

# Plot the metrics over the 10 trials
plt.figure(figsize=(12, 8))
plt.plot(precision_list, label="Precision", marker="o")
plt.plot(recall_list, label="Recall", marker="o")
plt.plot(f1_list, label="F1 Score", marker="o")
plt.title("Metrics over 10 Training Trials")
plt.xlabel("Trial")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
