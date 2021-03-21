import os

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from data_preprocessing import get_train_and_test_data
from keras import optimizers

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

xTrain, xTest, yTrain, yTest= get_train_and_test_data()

def prepare_training_and_test_data(nr_steps, nr_rows_to_drop, xTrain, yTrain, xTest, yTest):


    xTrain=xTrain[nr_rows_to_drop:]
    yTrain=yTrain[nr_rows_to_drop:]
    xTest=xTest[nr_rows_to_drop:]
    yTest=yTest[nr_rows_to_drop:]
    # indexes_for_y=[x for x in range(0, len(yTrain)) if not x % 7 is 0 ]
    # yTrain= np.delete(yTrain, [x for x in range(0, len(yTrain)) if not x % 7 is 0 ])
    # yTest= np.delete(yTest, [x for x in range(0, len(yTest)) if not x % 7 is 0 ])

    X=[]
    Y=[]
    X_test=[]
    Y_test=[]

    for i in range(0, len(xTrain)-nr_steps, nr_steps):
        X.append([xTrain[i:i+nr_steps]])
        Y.append(yTrain[i])
    for j in range(0, len(xTest)-nr_steps, nr_steps):
        X_test.append([xTest[j:j+nr_steps]])
        Y_test.append(yTest[j])
    X=np.array(X)
    X_test=np.array(X_test)
    Y=np.array(Y)
    Y_test=np.array(Y_test)

    X_= X.reshape(X.shape[0], X.shape[2], X.shape[3], order='F')
    Y_=Y.reshape(Y.shape[0], 1)
    X_test=X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], order='F')
    Y_test=Y_test.reshape(Y_test.shape[0], 1)

    return X_, Y_, X_test, Y_test


def train_model(X, Y, X_test, Y_test, nr_steps):
    n_batch = 14000
    n_epoch = 1000
    n_neurons = 30

    # design network
    model = Sequential()
    model.add(LSTM(units=n_neurons, return_sequences = True ,batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))

    model.add(Dropout(0.2))

    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=n_neurons, return_sequences=True))
    model.add(Dropout(0.2))


    model.add(Dense(1))
    optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss='mean_squared_error', optimizer=optimizer,  metrics=['acc'])

    for z in range(0, len(X)//4, n_batch):
        xChunk=X[z:z+n_batch]
        yChunk=Y[z:z+n_batch]

        model.fit(xChunk, yChunk, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()

    yhat = model.predict(X_test[0:n_batch, :, :], batch_size=n_batch, steps=nr_steps)
    for i in range(len(Y_test[0:n_batch])):
        print("Extected : "+ str(Y[i]) + " but actually: "+str(yhat[i][0]))
    error=1
    for j in range(len(Y_test[0:n_batch])):
        error= error*(abs(Y_test[j]-yhat[j][0])/yhat[j][0])*100
    print("error:   "+ str(error)+ "%")





X_, Y_, X_test, Y_test= prepare_training_and_test_data(nr_steps=7, nr_rows_to_drop=3, xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)
train_model(X=X_, Y=Y_, X_test=X_test, Y_test=Y_test, nr_steps= 7)
