# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
# https://keras.io/layers/core/
# gsettings set org.gnome.desktop.interface cursor-size 32
# open image
# open csv
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('tictactoe-data.csv')
print("sCORES:", Counter(df["score"]))

X = df.iloc[:, list(range(18)) + [-2]]
print(X)
X = np.asarray(X)

score = pd.get_dummies(df['score'])
print(score)
score = np.asarray(score)

X_train, X_test , y_train, y_test = train_test_split(X, score, test_size= 0.2)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_dim = X.shape[1]))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.3))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.3))
model.add(tf.keras.layers.Dense(score.shape[1], activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

model.fit(X_train, y_train, epochs=100, validation_data=[X_test, y_test])

loss , accuracy = model.evaluate(X_test, y_test )

print("Loss", loss)
print("Accuracy", accuracy)

model.save("tictacNET.h5")