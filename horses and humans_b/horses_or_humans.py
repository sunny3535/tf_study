# Question
#
# This task requires you to create a classifier for horses or humans using
# the provided data. Please make sure your final layer is a 1 neuron, activated by sigmoid as shown.
# Please note that the test will use images that are 300x300 with 3 bytes color depth so be sure to design your neural network accordingly

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - Horses Or Humans (Type B)
# val_loss: 0.51 (더 낮아도 안 좋고, 높아도 안 좋음!)
# val_acc: 관계없음
# =================================================== #
# =================================================== #



import tensorflow as tf
import urllib
import zipfile

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')
    local_zip = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/validation-horse-or-human/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest', 
    )

    validation_datagen = ImageDataGenerator(#Your Code here
        rescale=1. / 255,
    )

    TRAINING_DIR = 'tmp/horse-or-human/'
    VALIDATION_DIR = 'tmp/validation-horse-or-human/'       
    train_generator = train_datagen.flow_from_directory(
        #Your Code Here
            TRAINING_DIR, batch_size=32, target_size=(300, 300), class_mode='binary', 
        )

    validation_generator = validation_datagen.flow_from_directory(
        #Your Code Here
            VALIDATION_DIR, batch_size=32, target_size=(300, 300), class_mode='binary',
        )


    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D(2, 2), 
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2), 
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(), 
  
        Dropout(0.5),
        Dense(512, activation='relu'),
       
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(#Your Code Here# 
        optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    chk_location = "temp_checkpoint.ckpt"
    chk_point = ModelCheckpoint(filepath=chk_location, 
        save_weights_only=True, 
        save_best_only=True, 
        monitor='val_loss', 
        verbose=1)
    
    epochs=20
    model.fit(#Your Code Here#train_generator, 
        train_generator,
        validation_data=(validation_generator),
        epochs=epochs, 
        callbacks=[chk_point],
    )

    model.load_weights(chk_location)     
    # NOTE: If training is taking a very long time, you should consider setting the batch size appropriately on the generator, and the steps per epoch in the model.fit#
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-horses-or-humans-type-B.h5")