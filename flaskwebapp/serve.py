
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

def serve_model(x):
    filepath = './model/model_mnist_cnn.h5'
    model = keras.models.load_model(filepath)
    prediction = np.argmax(model.predict(x), axis=1)
    return prediction

def get_4_samples():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X = X.astype('float32')
    X /= 255
    return X_test[:4], X[:4]

def plot_4_digits(imgs):
    plt.figure(figsize=(2, 2))
    for i in range(4):
        img = imgs[i]
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('./fig/temp.png')
    return plt