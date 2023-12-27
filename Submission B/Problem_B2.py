# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load()
    # NORMALIZE YOUR IMAGE HERE
    train_images = train_images/255.0
    test_images = test_images/255.0
    
    # DEFINE YOUR MODEL HERE
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # End with 10 Neuron Dense, activated by softmax
    
    # Mendefinisikan Callbakcs untuk menghentikan training setelah akurasi mencapai 83.3%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83:
                print("\nTarget akurasi telah mencapai 83%, training dihentikan!")
                self.model.stop_training = True

        
    callbacks = myCallback()

    # COMPILE MODEL HERE
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 

    # TRAIN YOUR MODEL HERE
    model.fit(
        train_images, 
        train_labels, 
        epochs=10,
        validation_data=(test_images ,test_labels),
        callbacks=[callbacks]
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
