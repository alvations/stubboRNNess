# -*- coding: utf-8 -*-

import io
import sys

# Preprocessing intput files.
test_data = 'CWI-data/cwi_testing.txt'
# Preprocessing output files.
rnn_inputs_file = 'cwi_test.txt'

with io.open(test_data, 'r', encoding='utf8') as fin, \
io.open(rnn_inputs_file, 'w', encoding='utf8') as fout:    
    for line in fin:
        sent, word, idx  = line.strip().split('\t')
        fout.write(u"{} <s> {}\n".format(word, sent))
        
