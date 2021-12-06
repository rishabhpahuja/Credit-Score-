from Data_read import aus
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from ml_models import write_text_file
# x,y = aus()

# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

def nn(X_train,X_test,y_train,y_test):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", name="layer1"),
            layers.Dense(128, activation="relu", name="layer2"),
            layers.Dense(1, activation="sigmoid", name="layer3")
        ]
    )
    model.compile(optimizer='Adam',loss='binary_crossentropy')
    model.fit(X_train.to_numpy(),y_train.to_numpy(),batch_size=75,epochs=100)
    y_pred = model.predict(X_test.to_numpy())
    y_pred[y_pred<0.5]=0
    y_pred[y_pred>=0.5]=1

    score = metrics.accuracy_score(y_test.to_numpy(),y_pred)
    string = "Neural Network score: " + str(score)
    write_text_file(string)
    print("Neural Network score: ", score)
    acc = score
    return acc