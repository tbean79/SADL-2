#!/usr/bin/python3

import numpy as np
import sys
from gensim import utils
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=['POST'])
def hello_world():
    obj = request.get_json()
    print(obj, flush=True)
    #print(model.wv[obj["a"]], flush=True)
    history_vecs = []
    for item in obj["history"]:
        phrase_vec = []
        for word in remove_stopwords(item).split():
            word_lower = word.lower()
            if word_lower in model.wv:
                phrase_vec.append(model.wv[word_lower])
                #history_vecs.append(model.wv[word_lower])
        if len(phrase_vec) > 1:
            history_vecs.append((phrase_vec[0] - sum(phrase_vec[1:])) / len(phrase_vec))
        elif len(phrase_vec) == 1:
            history_vecs.append(phrase_vec[0])
    #average_vec = np.mean(history_vecs, axis=0)
    average_vec = sum(history_vecs)
    average_vec = average_vec / np.linalg.norm(average_vec)
    scores = {}
    for i in range(len(obj["results"])):
        phrase_vec = []
        for word in remove_stopwords(obj["results"][i]["expanded"]).split():
            word_lower = word.lower()
            if word_lower in model.wv:
                phrase_vec.append(model.wv[word_lower])
        if len(phrase_vec) > 1:
            average_phrase_vec = (phrase_vec[0] - sum(phrase_vec[1:])) / len(phrase_vec)
        elif len(phrase_vec) == 1:
            average_phrase_vec = phrase_vec[0]
        #average_phrase_vec = np.mean(phrase_vecs, axis=0)
        #average_phrase_vec = sum(phrase_vecs)
        average_phrase_vec = average_phrase_vec / np.linalg.norm(average_vec)
        sim_score = np.dot(average_vec, average_phrase_vec)
        scores[sim_score] = i
    print(scores, flush=True)
    ret_list = []
    for score in sorted(scores.keys(), reverse=True):
        ret_list.append(obj["results"][scores[score]])
    for i in range(len(obj["results"])):
        if i not in scores.values():
            ret_list.append(obj["results"][i])
    assert len(ret_list) == len(obj["results"])
    return ret_list, 200


def read_input(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            yield gensim.utils.simple_preprocess(line)

class MyCorpus:
    def __iter__(self):
        corpus_path = './long-abstracts_lang=en.ttl'
        #with gzip.open(corpus_path, 'rb') as f:
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)


model = Word2Vec.load('/tmp/gensim-model-5ci6irvl')
print('done with load', flush=True)

#if __name__=="__main__":
#    main()
