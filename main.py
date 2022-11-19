import numpy as np
from pickle import load
from keras.applications.xception import preprocess_input
from keras.utils import pad_sequences, load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

max_length = 34
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)
model = tf.keras.models.load_model('model_ic.h5')
with open("Pickle/variables.pkl", "rb") as variable_pickle:
    ixtoword, wordtoix = load(variable_pickle)


def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def generate_des(img_path):
    en_img = encode(img_path).reshape((1, 2048))
    x = plt.imread(img_path)
    plt.imshow(x)
    plt.show()
    return print(greedySearch(en_img))


file_path = r'D:\Media\pictures\281547532_556445102507664_4022151265332285771_n.jpg'
generate_des(file_path)
