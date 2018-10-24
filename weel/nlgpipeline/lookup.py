import collections
import csv
import itertools

import numpy
import torch

import fastText

from ..utils import random_vector, EOS, SOS, OOV, PAD, pad_all, to_batch_tensor, compute_mask

# start of sentence & end of sentence tokens



def compute_lookup(sequences, fastText_path, use_subwords=False):
    lookup = {}

    model = fastText.load_model(fastText_path)
    nb_dims = model.get_dimension()

    if use_subwords :
        add_pad = False
        vecs = {}
        for word in sequences :
            ngrams, subindices = model.get_subwords(word)
            lookup[word] = ngrams
            for w, idx in  zip(ngrams, subindices) :
                if w not in vecs :
                    vecs[w] = model.get_input_vector(idx)
        if not PAD in vecs:
            vecs[PAD] = None
            add_pad = True
        embedding_matrix = numpy.zeros((len(vecs),nb_dims))
        del model
        tmp_lookup = {}
        if add_pad :
            for i, subword in enumerate({s for w in lookup for s in lookup[w]}) :
                embedding_matrix[i + 1,:] = vecs[subword]
                tmp_lookup[subword] = i + 1
        else :
            for i, subword in enumerate({s for w in lookup for s in lookup[w]}) :
                embedding_matrix[i,:] = vecs[subword]
                tmp_lookup[subword] = i
        lookup = {
            word : [tmp_lookup[s] for s in lookup[word]]
            for word in lookup
        }
        if add_pad :
            lookup[PAD] = 0
        return lookup, embedding_matrix

    else :
        vecs = {
            w : model.get_word_vector(w)
            for s in sequences
            for w in s
        }
        if not EOS in vecs:
            vecs[EOS] = random_vector(nb_dims)
        if not SOS in vecs:
            vecs[SOS] = random_vector(nb_dims)
        if not PAD in vecs:
            vecs[PAD] = numpy.zeros((nb_dims,))
        embedding_matrix = numpy.zeros((len(vecs), nb_dims))
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

def make_batch(inputs, outputs, encoder_lookup, decoder_lookup):
    # convert to indices
    inputs = translate(inputs, encoder_lookup, use_subwords=True)
    outputs = translate(outputs, decoder_lookup)

    # sort on length
    data = list(zip(inputs, outputs))
    data.sort(key=lambda p: len(p[0]), reverse=True)
    inputs, outputs = zip(*data)

    # compute lengths for packed sequences
    inputs_lengths = torch.tensor(list(map(len, inputs)))
    max_target_length = max(map(len, outputs))
    # pad
    inputs = pad_all(inputs, padding_item=encoder_lookup[PAD])
    outputs = pad_all(outputs, padding_item=decoder_lookup[PAD])

    # compute mask
    outputs_mask = compute_mask(outputs)

    # pytorch-friendly format
    inputs = to_batch_tensor(inputs)
    outputs = to_batch_tensor(outputs)

    return inputs, inputs_lengths, outputs, outputs_mask, max_target_length
