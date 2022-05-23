import urllib.request
import zipfile
import os
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()

    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip'
    urllib.request.urlretrieve(url, 'rps-test-set.zip')
    local_zip = 'rps-test-set.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = "tmp/rps-test-set/"
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    training_generator = training_datagen.flow_from_directory(TRAINING_DIR, batch_size=32, target_size=(150, 150), class_mode='categorical',)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=32, target_size=(150, 150), class_mode='categorical', )

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
   
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
  
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
    
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.6),
 
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    chk_location = "temp_checkpoint.ckpt"
    chk_point = ModelCheckpoint(filepath=chk_location, 
        save_weights_only=True, 
        save_best_only=True, 
        monitor='val_loss', 
        verbose=1
    )
    epochs=25
    history = model.fit(training_generator, 
        validation_data=(validation_generator),
        epochs=epochs,
        callbacks=[chk_point],
        )

    model.load_weights(chk_location)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-rps.h5")