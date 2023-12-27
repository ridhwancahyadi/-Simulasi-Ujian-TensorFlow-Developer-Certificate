# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd

import numpy as np

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    
    # YOUR CODE HERE
    labels = bbc["category"].values.tolist()
    sentences = bbc["text"].values.tolist()

    # Split the dataset into training and validation sets
    training_size = int(len(sentences) * training_portion)
    training_sentences = sentences[:training_size]
    training_labels = labels[:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]
    
    # Tokenize sentences
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    # Convert sentences to sequences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    training_label_seq = np.array(
        label_tokenizer.texts_to_sequences(training_labels)
    )
    validation_label_seq = np.array(
        label_tokenizer.texts_to_sequences(validation_labels)
    )

    # Pad sequences
    training_padded = pad_sequences(
        training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    validation_padded = pad_sequences(
        validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])


    # Mendefinisikan Callbacks untuk menghentikan training setelah akurasi mencapai 91%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.92 and logs.get('val_accuracy') > 0.92:
                print("\nTarget akurasi telah mencapai 92%, training dihentikan!")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # Mengatur parameters dan optimizers
    from keras.optimizers import Adam
    adam = Adam(learning_rate=0.001)
    
    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    
    model.fit(
        training_padded,
        training_label_seq,
        epochs=150,
        validation_data=(
            validation_padded,
            validation_label_seq),
        callbacks=[callbacks],
        verbose=2)
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
