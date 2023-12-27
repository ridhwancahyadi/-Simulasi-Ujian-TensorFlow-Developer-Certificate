# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as f:
        datastore = json.load(f)

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
    
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    test_sentences = sentences[training_size:]
    test_labels = labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    
    # Convert sentences to sequences
    training_sequences = tokenizer.texts_to_sequences(train_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    
    # Pad sequences
    training_padded = pad_sequences(
        training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    test_padded = pad_sequences(
        test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Mendefinisikan Callbacks untuk menghentikan training setelah akurasi mencapai 81%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.81 and logs.get('val_accuracy') > 0.81:
                print("\nTarget akurasi telah mencapai 81%, training dihentikan!")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # Mengatur parameters dan optimizers
    from keras.optimizers import Adam
    adam = Adam(learning_rate=0.001)

    model.compile(
        loss='binary_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    
    model.fit(
        training_padded,
        train_labels,
        epochs=50,
        validation_data=(
            test_padded,
            test_labels),
        callbacks=[callbacks],
        verbose=2)
    
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
