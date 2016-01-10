# -*- coding: utf-8 -*-

import io
import sys

from pywsdlemmatizer import lemmatize_sentence
from nltk.tag import PerceptronTagger

tagger = PerceptronTagger()
pos_tag = tagger.tag

# Preprocessing intput files.
test_data = 'CWI-data/cwi_testing.txt'
# Preprocessing output files.
rnn_inputs_file = 'cwi_test.lemmatize.txt'

with io.open(test_data, 'r', encoding='utf8') as fin, \
io.open(rnn_inputs_file, 'w', encoding='utf8') as fout:    
    for line in fin:
        sent, word, idx  = line.strip().split('\t')
        sent = lemmatize_sentence(sent, neverstem=True)
        word = sent[int(idx)]
        sent = " ".join(sent)
        fout.write(u"{} <s> {}\n".format(word, sent))
        

# Preprocessing output files.
rnn_inputs_file = 'cwi_test.txt'

with io.open(test_data, 'r', encoding='utf8') as fin, \
io.open(rnn_inputs_file, 'w', encoding='utf8') as fout:
    for line in fin:
        sent, word, idx  = line.strip().split('\t')
        fout.write(u"{} <s> {}\n".format(word, sent))
