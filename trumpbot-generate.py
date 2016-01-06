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
from keras.models import Graph
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
import tweepy, time, sys

text = open('source.txt').read().lower()
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
lstm_hdim = 500
bridge_dim = 1000
dense_dim = 1500

sd_len = 12

batch_size = 128
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
        indexD = codedVector[idx * sd_len + iidx + 0:(idx + 1) * sd_len + iidx]
        i_D.append(indexD)

        vectorValD = [vmodel[myWord] for myWord in parsedWords[idx * sd_len + iidx + 0:(idx + 1) * sd_len + iidx]]
        x_D.append(vectorValD)
        y_D.append(one_hot(codedVector[(idx + 1) * sd_len + iidx]))

x_D = np.asarray(x_D)
y_D = np.asarray(y_D)
i_D = np.asarray(i_D)


layerNames = [
    'tdd1',
    'bn1',
    'lstm1',
    'bn2',
    'lstm2',
    'bn3',
    'lstm3',
    'bn4',
    'dropout1',
    'dense1',
    'denseelu1',
    'dropout2',
    'dense2',
    'densesm1',
]

# build the model: 2 stacked LSTM
print('shapes: ' + str((x_D.shape)))
print('Build model...')
model = Graph()
model.add_input(name='input', input_shape=(sd_len, input_dim))
model.add_node(TimeDistributedDense(input_dim=input_dim, output_dim=lstm_hdim, input_length=sd_len), name=layerNames[0], input='input')
model.add_node(BatchNormalization(), name=layerNames[1], input=layerNames[0])

model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True), name=layerNames[2] + 'left', input=layerNames[1])
model.add_node(BatchNormalization(), name=layerNames[3] + 'left', input=layerNames[2] + 'left')

model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True, go_backwards=True), name=layerNames[2] + 'right', input=layerNames[1])
model.add_node(BatchNormalization(), name=layerNames[3] + 'right', input=layerNames[2] + 'right')

model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=False), name=layerNames[6] + 'left', input=layerNames[3] + 'left')

model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=False, go_backwards=True), name=layerNames[6] + 'right', input=layerNames[3] + 'right')

model.add_node(BatchNormalization(), name=layerNames[7], inputs=[layerNames[6] + 'left', layerNames[6] + 'right'])
model.add_node(Dropout(0.2), name=layerNames[8], input=layerNames[7])

model.add_node(Dense(input_dim=bridge_dim, output_dim=dense_dim), name=layerNames[9], input=layerNames[8])
model.add_node(ELU(), name=layerNames[10], input=layerNames[9])
model.add_node(Dropout(0.2), name=layerNames[11], input=layerNames[10])

model.add_node(Dense(input_dim=dense_dim, output_dim=len(wordCoding)), name=layerNames[12], input=layerNames[11])
model.add_node(Activation('softmax'), name=layerNames[13], input=layerNames[12])
model.add_output(name='output1', input=layerNames[13])

model.compile(optimizer='rmsprop', loss={'output1':'categorical_crossentropy'})


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

seedLen = sd_len
seedSrc = i_D
outSentences = []
while len(outSentences) < 100:
    start_index = random.randint(0, len(seedSrc) - 1)
    sentence = seedSrc[start_index: start_index + 1]

    sentOutput = ''

    for iteration in range(500):
        vecsentence = []
        for vcode in sentence[0]:
            vecsentence.append(vmodel[codedWord[vcode]])
        vecsentence = np.reshape(vecsentence, (1, len(vecsentence), 300))
        preds = model.predict({'input':vecsentence}, verbose=0)['output1'][0]
        next_index = sample(preds, 0.2)
        if next_index in codedWord:
            next_char = codedWord[next_index]
            sentence = np.append(sentence[0][1:], [next_index]).reshape(np.asarray(sentence).shape)
            sentOutput += next_char + ' '
    print sentOutput
