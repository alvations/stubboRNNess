# -*- coding: utf-8 -*-

from __future__ import print_function

import io
import random
import sys

try:
	import cPickle as pickle
except:
	import pickle


from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import save, load

random.seed(0)

textfile, labelfile, embedding_size, gru_size, num_epochs = sys.argv[1:]

textfile = 'cwi_inputs.txt'
labelfile = 'cwi_labels.txt'
train_text, train_labels = [], []

with io.open(textfile, 'r', encoding='utf8') as txtfin, \
io.open(labelfile, 'r') as labelfin:
	for text, label in zip(txtfin, labelfin):
		train_text.append(text.strip())
		train_labels.append(int(label.strip()))

tokenizer = Tokenizer()
train_tokens = tokenizer.fit_transform(train_text)

layers = [Embedding(size=embedding_size, n_features=tokenizer.n_features),
          GatedRecurrent(size=gru_size),
          Dense(size=1, activation='sigmoid')]

model = RNN(layers=layers, cost='BinaryCrossEntropy')
model.fit(train_tokens, train_labels, n_epochs=int(num_epochs))
modelfile_name = 'stubborn_model.gridsearch.embedding{}.gru{}.epoch{}'.format(embedding_size, gru_size, num_epochs)
save(model, modelfile_name+ '.pkl')


