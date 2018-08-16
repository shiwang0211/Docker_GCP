
## sentiment
from script.sentiment_utils import *

def serve_model_sentiment(x):
    X = apply_phrase_model([x])
    X = np.array([dictionary.doc2idx(pad_trim_review(x), unknown_word_index=len(dictionary)) for x in X])
    prediction = lstm_model.predict(X)
    return str(prediction[0][0])

## mnist
def get_mnist_data():
    mnist_data = pickle.load( open( "mnist_data.p", "rb" ) )
    (X_train, y_train), (X_test, y_test) = mnist_data 
    X = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X = X.astype('float32')
    X /= 255
    return X_test, X

def plot_4_digits(imgs):
    plt.figure(figsize=(2, 2))
    for i in range(4):
        img = imgs[i]
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    return plt

def serve_model_mnist(random_index):
    raw_X, processed_X = get_mnist_data()
    raw_X, processed_X = raw_X[random_index], processed_X[random_index]
    plot = plot_4_digits(raw_X)
    prediction = np.argmax(mnist_model.predict(processed_X), axis=1)
    return plot, str(prediction)