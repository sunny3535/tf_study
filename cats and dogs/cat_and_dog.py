
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_name = 'cats_vs_dogs'

train_dataset = tfds.load(name=dataset_name, split='train[:80%]')
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')

def preprocess(data):

    x = data['image']
    y = data['label']
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.resize(x, size=(224, 224))
 
    return x, y

def solution_model():

    batch_size=32
    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    chk_location = "my_chk_point.ckpt"
    chk_point = ModelCheckpoint(filepath=chk_location, 
        save_weights_only=True,
        save_best_only=True, 
        monitor='val_loss', 
        verbose=1
    )
                                
    model.fit(train_data,
        validation_data=(valid_data),
        epochs=20,
        callbacks=[chk_point],
    )

    model.load_weights(chk_location)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-cats-vs-dogs.h5")
