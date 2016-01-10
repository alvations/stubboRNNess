# -*- coding: utf-8 -*-

import io
import sys

from pywsdlemmatizer import lemmatize_sentence

# Preprocessing intput files.
train_data = 'CWI-data/cwi_training.txt'
# Preprocessing output files.
rnn_inputs_file = 'cwi_inputs.txt'
rnn_labels_file = 'cwi_labels.txt'


with io.open(train_data, 'r', encoding='utf8') as fin, \
io.open(rnn_inputs_file, 'w', encoding='utf8') as inputsfile, \
io.open(rnn_labels_file, 'w', encoding='utf8') as labelsfile:
    for line in fin:
        sent, word, idx, label = line.strip().split('\t')
        inputsfile.write(u"{} <s> {}\n".format(word, sent))
        labelsfile.write(label + "\n")
        

rnn_inputs_file = 'cwi_inputs.lemmatized.txt'

with io.open(train_data, 'r', encoding='utf8') as fin, \
io.open(rnn_inputs_file, 'w', encoding='utf8') as inputsfile:
    for line in fin:
        sent, word, idx, label = line.strip().split('\t')
        sent = lemmatize_sentence(sent, neverstem=True)
        word = sent[int(idx)]
        sent = " ".join(sent)
        inputsfile.write(u"{} <s> {}\n".format(word, sent))

