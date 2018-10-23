import collections
import csv
import itertools

import numpy

import fastText

# start of sentence & end of sentence tokens
SOS = "<SOS>"

EOS = "<EOS>"

OOV = "<OOV>"


def compute_lookup(sequences, fastText_path, use_subwords=False):
    lookup = {}
    embedding_matrix = None

    model = fastText.load_model(fastText_path)

    if use_subwords :
        vecs = {}
        for word in sequences :
            ngrams, subindices = model.get_subwords(word)
            lookup[word] = ngrams
            for w, idx in  zip(ngrams, subindices) :
                if w not in vecs :
                    vecs[w] = model.get_input_vector(idx)
        embedding_matrix = numpy.zeros((len(vecs),model.get_dimension()))
        del model
        tmp_lookup = {}
        for i, subword in enumerate({s for w in lookup for s in lookup[w]}) :
            embedding_matrix[i,:] = vecs[subword]
            tmp_lookup[subword] = i
        lookup = {
            word : [tmp_lookup[s] for s in lookup[word]]
            for word in lookup
        }
        return lookup, embedding_matrix

    else :
        vecs = {
            w : model.get_word_vector(w)
            for s in sequences
            for w in s
        }
        if not EOS in vecs:
            vecs[EOS] = numpy.random.rand(model.get_dimension())
        if not SOS in vecs:
            vecs[SOS] = numpy.random.rand(model.get_dimension())
        embedding_matrix = numpy.zeros((len(vecs), model.get_dimension()))
        del model
        for i, s in enumerate(vecs) :
            embedding_matrix[i,:] = vecs[s]
            lookup[s] = i

        return lookup, embedding_matrix


def reverse_lookup(lookup) :
    return {lookup[item]:item for item in lookup}


def translate(sequences, lookup, use_subwords=False) :
    if use_subwords:
        return [lookup[word] for word in sequences]
    else:
        return [[lookup[item] for item in seq] for seq in sequences]
