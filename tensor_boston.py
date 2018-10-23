import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow import keras

from sklearn.model_selection import train_test_split

# reading the dataset


bos_df = pd.read_csv("/home/anjana/anjana/PYTHON/tensor/boston.csv")

# print the training set and tessting tests
# Shuffle the training set
X = bos_df.iloc[:, :13].values
y = bos_df.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0
)
print("Training set: {}".format(X_train.shape))
print("Testing set:  {}".format(X_test.shape))
print("Training label set:  {}".format(y_train.shape))
print("Testing label set:  {}".format(y_test.shape))
# Normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
train_data = (X_train - mean) / std
mean1 = X_test.mean(axis=0)
std1 = X_test.std(axis=0)
test_data = (X_test - mean1) / std1
print(train_data[0])  # First training sample, normalized
print(test_data[1])
# Create the model


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)
            ),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ]
    )
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model


model = build_model()
model.summary()
# Display training progress by printing a single dot for each completed epoch.


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


EPOCHS = 500

# Store training stats
history = model.fit(
    train_data,
    y_train,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()],
)


def plot_history(history):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [1000$]")
    plt.plot(
        history.epoch,
        np.array(history.history["mean_absolute_error"]),
        label="Train Loss",
    )
    plt.plot(
        history.epoch,
        np.array(history.history["val_mean_absolute_error"]),
        label="Val loss",
    )
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)
[loss, mae] = model.evaluate(test_data, y_test, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
# Predict
test_predictions = model.predict(test_data).flatten()
print(test_predictions)

