# -*- coding: UTF-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Masking
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.datasets.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import gensim as gs
import numpy as np
import random
import os
import sys

text = open('fuckface.txt').read().lower()
#fix misspellings, abbrvs and other trumpisms
text = text.replace(" - ", ", ")
text = text.replace(" t ", " to ")
text = text.replace(" st. louis ", " st louis ")
text = text.replace(" mr. ", " mister ")
text = text.replace(" mrs. ", " mistress ")
text = text.replace(" ms. ", " miss ")
text = text.replace(" feb. ", " february ")
text = text.replace(" dr. ", " doctor ")
text = text.replace(" gen. ", " general ")
text = text.replace(" gov. ", " governor ")
text = text.replace(" sen. ", " senator ")
text = text.replace(" sgt. ", " sergeant ")
text = text.replace(" rev. ", " reverend ")
text = text.replace(". no. ", ". number ")
text = text.replace(" jr. ", " junior ")

text = text.replace(".. ", ". ")
text = text.replace("penn.", "pennsylvania")
text = text.replace("wit-", "wit,")
text = text.replace("! ", " [xcm] ")
text = text.replace(". ", " [dot] ")
text = text.replace("? ", " [q] ")
text = text.replace(", ", " [comma] ")
text = text.replace(" $", " [dlr] ")
text = text.replace(": ", " [cln] ")
text = text.replace("; ", " [scln] ")
text = text.replace("% ", " [pcnt] ")
text = text.replace("   ", " ")
text = text.replace("  ", " ")
parsedWords = text.split(" ")
wordCoding = {}
codedWord = {}
codeNum = 1
codedVector = []
for word in parsedWords:
    if not word in wordCoding:
        wordCoding[word] = codeNum
        codedWord[codeNum] = word
        codeNum += 1
    codedVector.append(wordCoding[word])
print('corpus length:', len(wordCoding))
print('Vectorization...')

vmodel = gs.models.Word2Vec.load('trump2vec')

input_dim = 300
lstm_hdim = 600
sd_len = 8

batch_size = 256
sd_size = int(len(codedVector) / sd_len)

x_D = []# np.zeros((sc_size * sc_len, sc_len))
y_D = []# np.zeros((sc_size * sc_len, len(wordCoding)))
i_D = []

def one_hot(index):
    retVal = np.zeros((len(wordCoding)), dtype=np.bool)
    retVal[index] = 1
    return retVal

for idx in range(0, sd_size - 1):
    for iidx in range(0, sd_len - 1):
        vectorValD = [vmodel[myWord] for myWord in parsedWords[idx * sd_len + iidx + 0:(idx + 1) * sd_len + iidx]]
        x_D.append(vectorValD)
        y_D.append(one_hot(codedVector[(idx + 1) * sd_len + iidx]))

x_D = np.asarray(x_D)
y_D = np.asarray(y_D)
i_D = np.asarray(i_D)

# build the model: 2 stacked LSTM
print('shapes: ' + str((x_D.shape)))
print('Build model...')
model = Sequential()
model.add(TimeDistributedDense(input_dim=input_dim, output_dim=lstm_hdim, input_length=sd_len))
model.add(BatchNormalization())
model.add(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(input_dim=lstm_hdim, output_dim=lstm_hdim + 1000))
model.add(ELU())
model.add(Dropout(0.2))
model.add(Dense(input_dim=lstm_hdim + 1000, output_dim=len(wordCoding)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
def get_sentence(wordVec):
    sent = ''
    for wordVal in wordVec:
        sent += codedWord[wordVal] + ' '
    return sent

if os.path.isfile('tb-weights'):
    model.load_weights('tb-weights')

for iteration in range(0, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    for k in range(5):
        print('we at ' + str(k))
        model.fit(x_D, y_D, batch_size=batch_size, nb_epoch=1, show_accuracy=True)
    
    seedSelector = np.random.randint(0,3)
    seedSrc = i_D
    seedLen = sd_len
    model.save_weights('tb-weights', overwrite=True)
    start_index = random.randint(0, len(seedSrc) - 1)

    for diversity in [0.5, 1.0, 1.5]:
        print()
        print('----- diversity:', diversity)

        sentence = seedSrc[start_index: start_index + 1]
        strSentence = get_sentence(sentence[0])
        print('----- Generating with seed: "' + strSentence + '"')

        for iteration in range(500):
            vecsentence = []
            for vcode in sentence[0]:
                vecsentence.append(vmodel[codedWord[vcode]])
            vecsentence = np.reshape(vecsentence, (1, len(vecsentence), 300))
            preds = model.predict(vecsentence, verbose=0)[0]
            next_index = sample(preds, diversity)
            if next_index in codedWord:
                next_char = codedWord[next_index]
                sentence = np.append(sentence[0][1:], [next_index]).reshape(np.asarray(sentence).shape)

                sys.stdout.write(next_char + ' ')
                sys.stdout.flush()
        print()
