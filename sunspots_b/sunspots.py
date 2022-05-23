
import csv
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.layers import Dense, LSTM, Lambda, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2])) 
        time_step.append(int(row[0]))

    series = np.array(sunspots) 


    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    split_time = 3000


    time_train = time[:split_time]
    X_train = series[:split_time] 
    time_valid = time[split_time:]
    X_valid = series[split_time:] 

    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000


    train_set = windowed_dataset(X_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    validation_set = windowed_dataset(X_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(60, kernel_size=5,
		    padding="causal",
			activation="relu",
			input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
	    tf.keras.layers.LSTM(60, return_sequences=True),
	    tf.keras.layers.Dense(25, activation="relu"),
	    tf.keras.layers.Dense(15, activation="relu"),                                               

        tf.keras.layers.Dense(1)
    ])

    optimizer = SGD(lr=1e-5, momentum=0.9)
    loss= Huber()
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

    chk_location = 'temp_checkpoint.ckpt'
    chk_point = ModelCheckpoint(chk_location, 
        save_weights_only=True, 
        save_best_only=True, 
        monitor='val_mae',
        verbose=1
    )

    epochs=100
    history = model.fit(train_set, 
        validation_data=(validation_set), 
        epochs=epochs, 
        callbacks=[chk_point],
    )

    model.load_weights(chk_location)


    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF5-sunspots-type-B.h5")
