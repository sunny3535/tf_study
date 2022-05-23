
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_name = 'horses_or_humans'

train_dataset = tfds.load(name=dataset_name, split='train[:80%]')
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')
print(train_dataset)
def preprocess(data):
    x = data['image']
    y = data['label']
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.resize(x, size=(300, 300))
 
    return x, y


def solution_model():
    batch_size=32
    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    model = Sequential([
    
        Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D(2, 2), 
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
      
        Flatten(), 
       
        Dropout(0.5),
        Dense(512, activation='relu'),
       
        tf.keras.layers.Dense(2, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    chk_location = "temp_checkpoint.ckpt"
    chk_point = ModelCheckpoint(filepath=chk_location, 
        save_weights_only=True, 
        save_best_only=True, 
        monitor='val_loss', 
        verbose=1)


    epochs=20
    history = model.fit(train_data, 
	    validation_data=(valid_data),
        epochs=epochs, 
	    callbacks=[chk_point],
    )
					

    model.load_weights(chk_location)     


    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-horses-or-humans-type-A.h5")