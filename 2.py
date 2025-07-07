import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

data = pd.read_csv('heart.csv')


print(data.head())


print(data.describe())


print(data.isnull().any())

X = data.iloc[:, :13].values
y = data["target"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()


classifier.add(Dense(activation="relu", input_dim=13, units=8, kernel_initializer="uniform"))


classifier.add(Dense(activation="relu", units=14, kernel_initializer="uniform"))


classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


classifier.fit(X_train, y_train, batch_size=8, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


cm = confusion_matrix(y_test, y_pred)
print(cm)


accuracy = (cm[0][0] + cm[1][1]) / (cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print("Accuracy: {:.2f}%".format(accuracy * 100))
