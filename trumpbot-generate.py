# -*- coding: UTF-8 -*-
'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.datasets.data_utils import get_file
import numpy as np
import random
import os
import sys

text = open('source.txt').read().lower()
text = text.replace("!", " [xcm] ")
text = text.replace(".", " [dot] ")
text = text.replace(",", " [comma] ")
text = text.replace("?", " [q] ")
text = text.replace("$", " [dlr] ")
text = text.replace(":", " [cln] ")
text = text.replace(";", " [scln] ")
text = text.replace("%", " [pcnt] ")
text = text.replace("-", " [dsh]")
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

inputWords = np.asarray([[wordCoding['i'], wordCoding['love'], wordCoding['america'], wordCoding['because']]])
print('input shape' + str(inputWords.shape))
embed_dim = 1000
lstm_hdim = 2000
sd_len = 6

def one_hot(index):
    retVal = np.zeros((len(wordCoding)), dtype=np.bool)
    retVal[index] = 1
    return retVal

model = Sequential()
model.add(Embedding(input_dim=len(wordCoding), output_dim=lstm_hdim, mask_zero=True, input_length=sd_len))
model.add(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True))
model.add(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(input_dim=lstm_hdim, output_dim=lstm_hdim + 300))
model.add(ELU())
model.add(Dropout(0.1))
model.add(Dense(input_dim=lstm_hdim + 300, output_dim=lstm_hdim + 600))
model.add(ELU())
model.add(Dropout(0.1))
model.add(Dense(input_dim=lstm_hdim + 600, output_dim=len(wordCoding)))
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

for diversity in [0.5, 1.0, 1.5]:
    print()
    print('----- diversity:', diversity)

    sentence = inputWords
    strSentence = get_sentence(sentence[0])
    print('----- Generating with seed: "' + strSentence + '"')

    for iteration in range(500):

        preds = model.predict(sentence, verbose=0)[0]
        next_index = sample(preds, diversity)
        if next_index in codedWord:
            next_char = codedWord[next_index]
            sentence = np.append(sentence[0][1:], [next_index]).reshape(np.asarray(sentence).shape)

        sys.stdout.write(next_char + ' ')
        sys.stdout.flush()
    print()
