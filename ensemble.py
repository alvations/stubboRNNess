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

#textfile, labelfile, embedding_size, gru_size, num_epochs = sys.argv[1:]

textfile = 'cwi_inputs.lemmatized.txt'
labelfile = 'cwi_labels.txt'
train_text, train_labels = [], []

with io.open(textfile, 'r', encoding='utf8') as txtfin, \
io.open(labelfile, 'r') as labelfin:
    for text, label in zip(txtfin, labelfin):
        train_text.append(text.strip())
        train_labels.append(int(label.strip()))

tokenizer = pickle.load(open('cwi_inputs.lemmatized.txt-tokenizer.pkl', 'rb'))
best_hyperparams = [(10,10,10), (10,50,10), (50,200,10), (100,10,10), (200,20,10)]
train_tokens = tokenizer.fit_transform(train_text)

'''
# Test on train.
for embedding_size, gru_size, num_epochs in best_hyperparams:
    modelfile_name = 'stubborn_model.gridsearch.embedding{}.gru{}.epoch{}'.format(embedding_size, gru_size, num_epochs)
    model = load(modelfile_name + '.pkl')
    results = model.predict(train_tokens)

    count = 0
    for r, g in zip(results, train_labels):
        if int(r>=0.5) == int(g):
            count+=1
    results = 1.0 * count /len(train_labels)
    print (modelfile_name + '\t' + str(results))
'''

models = []
predictions =[]
for embedding_size, gru_size, num_epochs in best_hyperparams:
    modelfile_name = 'stubborn_model.gridsearch.embedding{}.gru{}.epoch{}'.format(embedding_size, gru_size, num_epochs)
    model = load(modelfile_name + '.pkl')
    models.append(model)


with io.open('ensemble.train', 'w') as fout:
    for instance, label in zip(train_tokens, train_labels):
        line = " ".join([str(model.predict([instance])[0][0]) for model in models])
        fout.write(unicode(line + " " + str(label) + '\n'))

testfile = 'cwi_test.lemmatized.txt'
test_text = []
with io.open(testfile, 'r', encoding='utf8') as fin:
    for text in fin:
        test_text.append(text.strip())

test_tokens = tokenizer.transform(test_text)
with io.open('ensemble.test', 'w') as fout:
    results = []
    for model in models:
        results.append(model.predict(test_tokens))
    for m1, m2, m3, m4, m5 in zip(*results):
        outline = unicode(" ".join(map(str, [m1[0], m2[0], m3[0], m4[0], m5[0]])) + '\n')
        #print (outline)
        fout.write(outline)

    

"""
# Test on train held-out (Unlemmatized)
stubborn_model.embedding10.gru10.epoch10	0.831111111111
stubborn_model.embedding10.gru50.epoch10	0.804444444444
stubborn_model.embedding50.gru200.epoch10	0.813333333333
stubborn_model.embedding100.gru10.epoch10	0.817777777778
stubborn_model.embedding200.gru20.epoch10	0.831111111111

# Test on train (Unlemmatized)
stubborn_model.gridsearch.embedding10.gru10.epoch10	0.801513128616
stubborn_model.gridsearch.embedding10.gru50.epoch10	0.805073431242
stubborn_model.gridsearch.embedding50.gru200.epoch10	0.803293279929
stubborn_model.gridsearch.embedding100.gru10.epoch10	0.813974187806
stubborn_model.gridsearch.embedding200.gru20.epoch10	0.805963506898


# Test on train.lemmatized.
stubborn_model.gridsearch.embedding10.gru10.epoch10	0.807298620383
stubborn_model.gridsearch.embedding10.gru50.epoch10	0.809078771696
stubborn_model.gridsearch.embedding50.gru200.epoch10	0.804628393413
stubborn_model.gridsearch.embedding100.gru10.epoch10	0.878504672897
stubborn_model.gridsearch.embedding200.gru20.epoch10	0.877169559413

"""
