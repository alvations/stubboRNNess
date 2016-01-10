# -*- coding: utf-8 -*-

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

testfile, modelfile, tokenizerfile = sys.argv[1:]

test_text = []
with io.open(testfile, 'r', encoding='utf8') as fin:
    for text in fin:
        test_text.append(text.strip())

tokenizer = pickle.load(open(tokenizerfile, 'rb'))
model = load(modelfile)
results = model.predict(tokenizer.transform(test_text))

for r in results:
    print (int(r>=0.5))
