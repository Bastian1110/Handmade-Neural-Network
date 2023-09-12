"""
Neural Network to analize Wine Quality calssificator model
By: Sebastian Mora (@Bastian1110)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2


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


def regenerate_datasets(X, y):
    # Split the data into training, validation, and test sets
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

    return X_train, X_val, X_test, y_train, y_val, y_test


train_confusion = np.zeros((3, 3))
val_confusion = np.zeros((3, 3))
test_confusion = np.zeros((3, 3))


train_errors = {"bad": [], "regular": [], "good": []}
val_errors = {"bad": [], "regular": [], "good": []}
test_errors = {"bad": [], "regular": [], "good": []}

train_accuracies = []
val_accuracies = []
test_accuracies = []


# Model training and evaluation

epochs = 250
trials = 10


for i in range(trials):
    print(f"Trial : {i}")
    X_train, X_val, X_test, y_train, y_val, y_test = regenerate_datasets(X, y)
    # Convert the class labels to one-hot encoded vectors for neural network training
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

    # Build a neural network model
    from keras.optimizers import Adam

    optimizer = Adam(learning_rate=0.0001)

    model = Sequential()
    model.add(
        Dense(
            128,
            activation="relu",
            input_shape=(X_train.shape[1],),
            kernel_regularizer=l2(0.01),
        )
    )
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.012)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))  # 3 output units for 3 classes
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model using the training data
    model.fit(
        X_train,
        y_train_onehot,
        validation_data=(X_val, to_categorical(y_val)),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )

    y_pred = model.predict(X_train)
    y_pred_classes = np.argmax(y_pred, axis=1)
    train_report = classification_report(
        y_train,
        y_pred_classes,
        target_names=["bad", "regular", "good"],
        output_dict=True,
    )
    train_confusion += confusion_matrix(y_train, y_pred_classes)

    for category in ["bad", "regular", "good"]:
        train_errors[category].append(1 - train_report[category]["f1-score"])

    # Errors for validation set
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    val_report = classification_report(
        y_val, y_pred_classes, target_names=["bad", "regular", "good"], output_dict=True
    )

    for category in ["bad", "regular", "good"]:
        val_errors[category].append(1 - val_report[category]["f1-score"])
    val_confusion += confusion_matrix(y_val, y_pred_classes)

    # Errors for test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_report = classification_report(
        y_test,
        y_pred_classes,
        target_names=["bad", "regular", "good"],
        output_dict=True,
    )

    for category in ["bad", "regular", "good"]:
        test_errors[category].append(1 - test_report[category]["f1-score"])

    train_accuracies.append(train_report["accuracy"])
    val_accuracies.append(val_report["accuracy"])
    test_accuracies.append(test_report["accuracy"])

    # Para el conjunto de entrenamiento
    test_confusion += confusion_matrix(y_test, y_pred_classes)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))


def plot_accuracies(train_acc, val_acc, test_acc, title):
    x_axis = range(1, trials + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(x_axis, train_acc, label="Training Accuracy")
    plt.plot(x_axis, val_acc, label="Validation Accuracy")
    plt.plot(x_axis, test_acc, label="Test Accuracy")

    plt.title(title)
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import seaborn as sns

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.heatmap(train_confusion, annot=True, fmt="g", cmap="Blues", ax=ax[0])
ax[0].set_title("Train Confusion Matrix")
ax[0].set_xlabel("Predicted labels")
ax[0].set_ylabel("True labels")

sns.heatmap(val_confusion, annot=True, fmt="g", cmap="Blues", ax=ax[1])
ax[1].set_title("Validation Confusion Matrix")
ax[1].set_xlabel("Predicted labels")
ax[1].set_ylabel("True labels")

sns.heatmap(test_confusion, annot=True, fmt="g", cmap="Blues", ax=ax[2])
ax[2].set_title("Test Confusion Matrix")
ax[2].set_xlabel("Predicted labels")
ax[2].set_ylabel("True labels")

plt.tight_layout()
plt.show()


plot_accuracies(
    train_accuracies, val_accuracies, test_accuracies, "Accuracy across Trials"
)


labels = ["bad", "regular", "good"]

train_means = [np.mean(train_errors[category]) for category in labels]
val_means = [np.mean(val_errors[category]) for category in labels]
test_means = [np.mean(test_errors[category]) for category in labels]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, train_means, width, label="Train", color="b")
rects2 = ax.bar(x, val_means, width, label="Validation", color="y")
rects3 = ax.bar(x + width, test_means, width, label="Test", color="r")

ax.set_xlabel("Categories")
ax.set_ylabel("Error")
ax.set_title("Errors by dataset and wine category")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.show()
