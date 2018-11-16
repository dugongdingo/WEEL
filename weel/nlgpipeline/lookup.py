import collections
import csv
import itertools

import numpy
import torch

import fastText

from ..utils import random_vector, EOS, SOS, OOV, PAD, pad_all, to_batch_tensor, compute_mask, lmap

# start of sentence & end of sentence tokens



def compute_lookup(sequences, fastText_path, use_subwords=False, trim_threshold=None):
    lookup = {}

    model = fastText.load_model(fastText_path)
    nb_dims = model.get_dimension()

    if use_subwords :
        add_pad = False
        vecs = collections.OrderedDict()
        for word in sequences :
            ngrams, subindices = model.get_subwords(word)
            lookup[word] = ngrams
            for w, idx in  zip(ngrams, subindices) :
                if w not in vecs :
                    vecs[w] = model.get_input_vector(idx)
        if not OOV in vecs:
            vecs[OOV] = random_vector(nb_dims)
        if not EOS in vecs:
            vecs[EOS] = random_vector(nb_dims)
        if not SOS in vecs:
            vecs[SOS] = random_vector(nb_dims)
        if not PAD in vecs:
            vecs[PAD] = random_vector(nb_dims)
        embedding_matrix = numpy.zeros((len(vecs),nb_dims))
        del model
        tmp_lookup = {}
        for i, subword in enumerate([PAD,OOV,EOS,SOS] + list({s for w in lookup for s in lookup[w]})) :
            embedding_matrix[i,:] = vecs[subword]
            if subword not in {PAD,OOV,EOS,SOS} :
                tmp_lookup[subword] = i
        lookup = {
            word : [tmp_lookup[s] for s in lookup[word]]
            for word in lookup
        }
        lookup[PAD] = 0
        lookup[OOV] = 1
        lookup[EOS] = 2
        lookup[SOS] = 3
    else :
        vecs = collections.OrderedDict()
        vecs.update({
            w : model.get_word_vector(w)
            for s in sequences
            for w in s
        })
        if not OOV in vecs:
            vecs[OOV] = random_vector(nb_dims)
        if not EOS in vecs:
            vecs[EOS] = random_vector(nb_dims)
        if not SOS in vecs:
            vecs[SOS] = random_vector(nb_dims)
        if not PAD in vecs:
            vecs[PAD] = random_vector(nb_dims)
        embedding_matrix = numpy.zeros((len(vecs) + 1, nb_dims))
        del model

        vecs.move_to_end(SOS, last=False)
        vecs.move_to_end(EOS, last=False)
        vecs.move_to_end(OOV, last=False)
        vecs.move_to_end(PAD, last=False)
        for i, s in enumerate(vecs) :
            embedding_matrix[i,:] = vecs[s]
            lookup[s] = i
    return lookup, embedding_matrix


def reverse_lookup(lookup) :
    rlookup = {lookup[item]:item for item in lookup}
    rlookup[lookup[OOV]] = OOV
    return rlookup


def translate(sequences, lookup, use_subwords=False) :
    if use_subwords:
        return [lookup[word] for word in sequences]
    else:
        return [[lookup[item] for item in seq] for seq in sequences]


def make_batch(inputs, outputs, encoder_lookup, decoder_lookup, hollistic_lookup):
    # convert to indices
    hollistic_indices = translate([[i] for i in inputs], hollistic_lookup)
    inputs = translate(inputs, encoder_lookup, use_subwords=True)
    outputs = translate(outputs, decoder_lookup)


    # sort on length
    data = list(zip(inputs, outputs, hollistic_indices))
    data.sort(key=lambda p: len(p[0]), reverse=True)
    inputs, outputs, hollistic_indices = zip(*data)

    # compute lengths for packed sequences
    inputs_lengths = torch.tensor(lmap(len, inputs))
    max_target_length = max(map(len, outputs))
    # pad
    inputs = pad_all(inputs, padding_item=encoder_lookup[PAD])
    outputs = pad_all(outputs, padding_item=decoder_lookup[PAD])

    # compute mask
    outputs_mask = compute_mask(outputs, padding_item=decoder_lookup[PAD])

    # pytorch-friendly format
    inputs = to_batch_tensor(inputs)
    outputs = to_batch_tensor(outputs)
    hollistic_indices = to_batch_tensor(hollistic_indices)

    return inputs, inputs_lengths, outputs, outputs_mask, hollistic_indices, max_target_length


def mock_lookup(seqs) :
    lu = {c:i+1 for i,c in enumerate({c for seq in seqs for c in seq})}
    lu[PAD] = 0
    return lu, None
