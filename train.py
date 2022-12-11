#!/usr/bin/python3

import tempfile
import numpy as np
import sys
from gensim.test.utils import datapath
from gensim import utils
from gensim.models.word2vec import Word2Vec



def read_input(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            yield gensim.utils.simple_preprocess(line)

class MyCorpus:
    def __iter__(self):
        corpus_path = './long-abstracts_lang=en.ttl'
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)

def init_model():
    corpus = MyCorpus()
    word_model = Word2Vec(sentences=corpus)
    return word_model

def main():
    model = init_model()
    with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
        temporary_filepath = tmp.name
        model.save(temporary_filepath)

if __name__=="__main__":
    main()
