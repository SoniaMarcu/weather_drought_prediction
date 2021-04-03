import os

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers
import pandas as pd
import random
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


# xTrain, xTest, yTrain, yTest= get_train_and_test_data()
# xTrain=pd.read_csv("./xTrain.csv")
# yTrain=pd.read_csv("./yTrain.csv")
# xTest=pd.read_csv("./xTest.csv")
# yTest=pd.read_csv("./yTest.csv")

def remove_many_fluffing_zeros():
    yTrain = pd.read_csv('yTrain.csv', sep=',')
    keep_prob = 0.1
    cont = 0
    throw_array = [0, 1, 2, 3]
    print(yTrain.shape)
    for i in range(10, yTrain.shape[0], 7):
        if yTrain.iloc[i].to_numpy()[0] == 0:
            if random.random() > keep_prob:
                throw_array.extend([x for x in range(i - 6, i + 1)])
    yTrain = yTrain.drop(throw_array)
    print(throw_array[0:1000])
    print(yTrain.shape[0])
    yTrain.to_csv('yTrainCleansed.csv', index=False)
    xTrain = pd.read_csv('xTrain.csv', sep=',')
    xTrain = xTrain.drop(throw_array)
    xTrain.to_csv('xTrainCleansed.csv', index=False)


def summarize_performance(model, xChunk, yChunk, batch_counter):
    loss, acc = model.evaluate(xChunk, yChunk, verbose=0)
    print('>Accuracy: ' + str(acc * 100) + ', Loss: ' + str(loss) + ', Batch Number: ' + str(batch_counter))


def create_model(elems_in_batch):
    model = Sequential()
    model.add(LSTM(units=100, batch_input_shape=(elems_in_batch, 1, 49), return_sequences=True, activation='relu'))
    # Nu se poate salva modelul daca e stateful. Adica eu nu am reusit
    # model.add(LSTM(units=100,batch_input_shape=(elems_in_batch,1,49),return_sequences=True,activation='relu',stateful=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer, metrics=['acc'])
    return model


def train_model():
    yTrain = pd.read_csv('yTrainCleansed.csv', sep=',')
    elems_in_batch = 10000
    model = create_model(elems_in_batch)
    epochs = 20
    for epoch in range(epochs):
        contor = 0
        for df in pd.read_csv('xTrainCleansed.csv', sep=',', skiprows=contor, chunksize=7 * elems_in_batch):
            if df.shape[0] == 7 * elems_in_batch:
                xChunk = np.reshape(df.to_numpy(), (elems_in_batch, 1, 49))
                yChunk = yTrain.iloc[contor + 6].to_numpy() * 20
                for i in range(1, elems_in_batch):
                    yChunk = np.append(yChunk, yTrain.iloc[contor + 6 + i * 7].to_numpy() * 20, axis=0)
                contor += 7 * elems_in_batch
                yChunk = np.reshape(yChunk, (elems_in_batch, 1))
                model.train_on_batch(xChunk, yChunk)
                model.reset_states()
                summarize_performance(model, xChunk, yChunk, (contor - 3) // (elems_in_batch * 7))
        print("End of epoch " + str(epoch))

    model.save('model1')


# remove_many_fluffing_zeros()
train_model()
