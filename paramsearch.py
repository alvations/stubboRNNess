# -*- coding: utf-8 -*-

from __future__ import print_function

import io
import random
import sys
from itertools import product

try:
	import cPickle as pickle
except:
	import pickle

from sklearn import cross_validation

from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import save, load

random.seed(0)

textfile, labelfile = sys.argv[1:]

train_text, train_labels = [], []

with io.open(textfile, 'r', encoding='utf8') as txtfin, \
io.open(labelfile, 'r') as labelfin:
	for text, label in zip(txtfin, labelfin):
		train_text.append(text.strip())
		train_labels.append(int(label.strip()))

tokenizer = Tokenizer()
train_tokens = tokenizer.fit_transform(train_text)

embedding_sizes = [10, 20, 50, 100, 200, 1000]
gru_sizes = [10, 20, 50, 100, 200, 1000]
epochs = [1, 3, 5, 7, 10]

for embedding_size, gru_size, num_epochs in product(embedding_sizes, gru_sizes, epochs):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_text, train_labels, test_size=0.1, random_state=0)

    layers = [
        Embedding(size=embedding_size, n_features=tokenizer.n_features),
        GatedRecurrent(size=gru_size),
        Dense(size=1, activation='sigmoid')
    ]

    model = RNN(layers=layers, cost='BinaryCrossEntropy')
    model.fit(train_tokens, train_labels, n_epochs=int(num_epochs))

    modelfile_name = 'stubborn_model.embedding{}.gru{}.epoch{}'.format(embedding_size, gru_size, num_epochs)

    save(model, modelfile_name+ '.pkl')
    pickle.dump(tokenizer, open(modelfile_name + '-tokenizer.pkl', 'wb'))
    
    results = model.predict(tokenizer.transform(X_test))

    count = 0
    for r, g in zip(results, y_test):
	    if int(r>=0.5) == int(g):
		    count+=1
    results = 1.0 * count /len(y_test)
    print (modelfile_name + '\t' + str(results))
