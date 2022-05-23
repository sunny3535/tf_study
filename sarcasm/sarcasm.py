
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    with open('sarcasm.json') as f:
	    datas = json.load(f)
 
    for data in datas:
	    sentences.append(data['headline'])
	    labels.append(data['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

    tokenizer.fit_on_texts(train_sentences)
    for key, value in tokenizer.word_index.items():
	    print('{}  \t======>\t {}'.format(key, value))
	    if value == 25:
		    break
    
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)
      
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)

    model = tf.keras.Sequential([
        
        Embedding(vocab_size, embedding_dim, input_length=max_length),
	    Bidirectional(LSTM(64, return_sequences=True)),
	    Bidirectional(LSTM(64)),
	    Dense(32, activation='relu'),
	    Dense(16, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    chk_location = 'my_chk_point.ckpt'
    chk_point = ModelCheckpoint(chk_location, 
	    save_weights_only=True, 
	    save_best_only=True, 
	    monitor='val_loss',
	    verbose=1
    )
						
    epochs=10						
    history = model.fit(train_padded, train_labels, 
	    validation_data=(validation_padded, validation_labels),
	    callbacks=[chk_point],
	    epochs=epochs
    )

    model.load_weights(chk_location)


    return model


if __name__ == '__main__':
    model = solution_model()
    model.save("TF4-sarcasm.h5")