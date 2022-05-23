
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint



def solution_model():
    mnist = tf.keras.datasets.mnist

    (X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    model = Sequential([

	    Flatten(input_shape=(28, 28)),
	    Dense(1024, activation='relu'),
	    Dense(256, activation='relu'),
	    Dense(255, activation='relu'),
        Dense(255, activation='relu'),
        Dense(128, activation='relu'),
        Dense(16, activation='relu'),

	    Dense(10, activation='softmax'),
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    chk_location = "temp_checkpoint.ckpt"
    chk_point = ModelCheckpoint(filepath=chk_location, 
	    save_weights_only=True, 
	    save_best_only=True,  
	    monitor='val_loss', 
	    verbose=1
    )
						
						
    model.fit(X_train, Y_train,
	    validation_data=(X_valid, Y_valid),
	    epochs=20,
	    callbacks=[chk_point],
    )

    model.load_weights(chk_location)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-mnist.h5")