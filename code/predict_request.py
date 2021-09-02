import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

import parameter

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)


def generate_request():

    u = 30
    sig = math.sqrt(10)
    x = np.linspace(u - 3*sig, u + 3*sig, parameter.initial_request_num)
    data = np.exp(-(x - u) ** 2 / (2 * sig ** 2))/(math.sqrt(2 * math.pi) * sig)
    data = np.concatenate([data, data, data])

    with open("data/initial_request.csv", "w", newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["request_num"])
        for i in range(data.shape[0]):
            data[i] = int(data[i] * 6000 + 100)
            noise = np.random.uniform(-10, 10, 3 * parameter.initial_request_num)
            csvWriter.writerow([int(data[i] + noise[i])])
        csvFile.close()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


def lstm(look_back):

    generate_request()

    dataframe = pd.read_csv("data/initial_request.csv", engine="python")
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    return testPredict



def get_predict():
    data = lstm(1)

    with open("data/predict_request.csv", "w", newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["predict_request"])
        for i in range(data.shape[0]):
            csvWriter.writerow([int(data[i])])
        csvFile.close()
