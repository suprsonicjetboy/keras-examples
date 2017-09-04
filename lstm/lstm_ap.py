"""A very simple LSTM example.
Predict the positive integers.
"""
import urllib, urllib.request
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.backend.tensorflow_backend import set_session
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.45
    ),
    device_count = {
        'GPU': 0
    }
)
set_session(tf.Session(config=config))


class PredictionLSTM :

    def __init__(self):
        self.look_back = 1
        self.units = 128
        self.epochs = 10
        self.batch_size = 1


    def create_dataset(self, dataset, look_back=1):
        x, y = [], []
        for i in range(len(dataset) - look_back):
            a = i + look_back
            x.append(dataset[i:a, 0])
            y.append(dataset[a, 0])

        return np.array(x), np.array(y)


    def create_model(self):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model


    def train(self, x, y):
        model = self.create_model()
        model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

        return model


if __name__ == "__main__":
    START_TIME = time.time()
    SERIES_LENGTH = 1000

    # Prepare dataset
    dataset = np.arange(1, SERIES_LENGTH+1, 1).reshape(SERIES_LENGTH, 1).astype(np.float)

    # Transform
    scaler = preprocessing.MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # Split dataset into train and test subsets
    train_dataset = dataset[0:int(len(dataset)*0.8), :]
    test_dataset =  dataset[len(train_dataset):len(dataset), :]



    # LSTM
    prediction_ltsm = PredictionLSTM()

    # Create train dataset
    train_x, train_y = prediction_ltsm.create_dataset(train_dataset, prediction_ltsm.look_back)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    # Create test dataset
    test_x, test_y  = prediction_ltsm.create_dataset(test_dataset, prediction_ltsm.look_back)
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))


    # Create and fit the LSTM network
    model = prediction_ltsm.train(train_x, train_y)
    print(model.summary())


    # Predict train dataset
    train_prediction = model.predict(train_x)
    train_prediction = scaler.inverse_transform(train_prediction)
    train_y = scaler.inverse_transform([train_y])

    # Predict test dataset
    test_prediction = model.predict(test_x)
    test_prediction = scaler.inverse_transform(test_prediction)
    test_y = scaler.inverse_transform([test_y])


    # Calculate RMSE(Root Mean Squared Error)
    train_score = math.sqrt(mean_squared_error(train_y[0], train_prediction[:, 0]))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_prediction[:, 0]))
    print("\nTrain Score: {0:.3f} RMSE".format(train_score))
    print("Test  Score: {0:.3f} RMSE".format(test_score))


    # Predict the next value using the latest data
    latest_x = np.array([test_dataset[-prediction_ltsm.look_back:]])
    latest_x = np.reshape(latest_x, (latest_x.shape[0], 1, latest_x.shape[1]))
    next_prediction = model.predict(latest_x)
    next_prediction = scaler.inverse_transform(next_prediction)
    print("\nNext prediction: {0:.2f}".format(list(next_prediction)[0][0]), "\n"*2)

    print("Time: {0:.1f}sec".format(time.time() - START_TIME))


    # Draw a figure
    placeholder = np.append(dataset, np.zeros((1, dataset.shape[1])), axis=0)
    placeholder[:, :] = np.nan

    correct_dataset_plt = scaler.inverse_transform(dataset)

    train_plt = np.copy(placeholder)
    train_plt[prediction_ltsm.look_back:len(train_prediction)+prediction_ltsm.look_back, :] = train_prediction

    test_plt = np.copy(placeholder)
    test_plt[len(train_prediction)+(prediction_ltsm.look_back*2):len(dataset), :] = test_prediction

    nest_plt = np.copy(placeholder)
    nest_plt[len(placeholder)-2:len(placeholder), :] = np.append(test_prediction[-1], next_prediction.reshape(1)).reshape(2, 1)

    plt.plot(correct_dataset_plt, label='Correct prediction')
    plt.plot(train_plt, label='Train')
    plt.plot(test_plt, label='Test')
    plt.plot(nest_plt, label='Next prediction', c='r')
    plt.legend() 
    plt.show()
    