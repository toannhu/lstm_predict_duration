from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load dataset
dataset = read_csv('format.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
re_phones = [
    u'sil',

    u'b',
    u'd',
    u'g',
    u'p',
    u't',
    u'k',
    u'dx',
    u'q',

    u'bcl',
    u'dcl',
    u'gcl',
    u'pcl',
    u'tcl',
    u'kcl',

    u'dcl',
    u'tcl',

    u'jh',
    u'ch',

    u's',
    u'sh',
    u'z',
    u'zh',
    u'f',
    u'th',
    u'v',
    u'dh',

    u'm',
    u'n',
    u'ng',
    u'em',
    u'en',
    u'eng',
    u'nx',

    u'l',
    u'r',
    u'w',
    u'y',
    u'hh',
    u'hv',
    u'el',

    u'iy',
    u'ih',
    u'eh',
    u'ey',
    u'ae',
    u'aa',
    u'aw',
    u'ay',
    u'ah',
    u'ao',
    u'oy',
    u'ow',
    u'uh',
    u'uw',
    u'ux',
    u'er',
    u'ax',
    u'ix',
    u'axr',
    u'ax-h',

    u'pau',
    u'epi',
    u'1',
    u'2'
]

encoder.fit(re_phones)
for i in range(0,4):
    values[:,i] = encoder.transform(values[:,i])

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

# frame as supervised learning
df = DataFrame(values)
print(df.head())

# split into train and test sets
values = df.values
n_train_samples = 50000
train = values[:n_train_samples, :]
test = values[n_train_samples:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=True)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
# model.save("model.h5")

from keras.models import load_model
model = load_model("model.h5")

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((test_X[:, 0:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
print("Forecast value: ")
print(inv_yhat[0:10,:])

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, 0:], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
print("True value: ")
print(inv_y[0:10,:])

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
