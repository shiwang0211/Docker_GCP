import numpy as np
import tensorflow as tf
import keras
import pickle
import spacy
import gensim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
nlp = spacy.load('en')
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import corpora
bigram_model = Phraser.load('./model/phrase_model_bigram')
trigram_model = Phraser.load('./model/phrase_model_trigram')
dictionary = corpora.Dictionary.load('./model/dictionary')
lstm_model = keras.models.load_model('./model/yelp_lstm_sentiment.h5')
mnist_model = keras.models.load_model('./model/model_mnist_cnn.h5')