from Data_read import aus
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

x,y = aus()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
score = metrics.accuracy_score(y_test,y_pred)
print(score)

