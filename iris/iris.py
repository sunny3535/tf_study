
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

def preprocess(data):
	x = data['features']
	y = data['label']
	y = tf.one_hot(y, 3)
	return x, y

def solution_model():
	#train_dataset = data.map(preprocess).batch(10)
	batch_size=10
	train_data = train_dataset.map(preprocess).batch(batch_size)
	valid_data = valid_dataset.map(preprocess).batch(batch_size)

	model = tf.keras.models.Sequential([

		Dense(512, activation='relu', input_shape=(4,)),
		Dense(256, activation='relu'),
		Dense(128, activation='relu'),
		Dense(64, activation='relu'),
		Dense(32, activation='relu'),
		Dense(16, activation='relu'),
		Dense(8, activation='relu'),
		Dense(4, activation='relu'),

		Dense(3, activation='softmax'),
	])

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

	chk_location = "temp_checkpoint.ckpt"
	chk_point = ModelCheckpoint(filepath=chk_location, 
		save_weights_only=True,
		save_best_only=True,
		monitor='val_loss',
		verbose=1
	)
						
						
	history = model.fit(train_data,
		validation_data=(valid_data),
		epochs=20,
		callbacks=[chk_point], 
	)

	model.load_weights(chk_location)

	return model

if __name__ == '__main__':
	model = solution_model()
	model.save("TF2-iris.h5")