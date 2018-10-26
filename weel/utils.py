import csv
import datetime
from itertools import chain, islice, repeat, zip_longest

import numpy
import nltk.tokenize
import torch

from .settings import DEVICE, MAX_LENGTH, BATCH_SIZE

SOS = "<SOS>"

EOS = "<EOS>"

OOV = "<OOV>"

PAD = "<PAD>"


def to_tensor(seq, device=DEVICE) :
    return torch.tensor(seq, dtype=torch.long, device=device).view(-1, 1)

def to_device(*tensors, device=DEVICE):
    return [t.to(device) for t in tensors]

def to_sentence(raw_str):
    return [SOS] + nltk.tokenize.word_tokenize(raw_str) + [EOS]


def pad_one(sequence, max_length=MAX_LENGTH, padding_item=PAD) :
    return list(islice(chain(sequence, repeat(padding_item)), max_length))


def pad_all(sequences, max_length=None, padding_item=PAD) :
    if max_length is None :
        return  list(zip_longest(*sequences, fillvalue=padding_item))
    else :
        return [
            pad_one(s, max_length=max_length, padding_item=padding_item)
            for s in sequences
        ]


def to_batch_tensor(padded_sequences, device=DEVICE) :
    return torch.LongTensor(padded_sequences).to(device)


def compute_mask(padded_sequences, padding_item=PAD, device=DEVICE) :
    pad_list = [
        int(elem != padding_item)
        for seq in padded_sequences
        for elem in seq
    ]
    return torch.ByteTensor(pad_list).to(device)


def random_vector(nb_dims):
     return numpy.random.rand(nb_dims)


def print_now(*line) :
    """
    utility function: prepend timestamp to std output
    """
    print(datetime.datetime.now(), ":", *line)


def read_parsed_data_file(datafile) :
    """
    create (word, definition) pairs from parsed wiktionary/ WordNet
    """
    with open(datafile, "r") as istr :
        istr.readline() #skip header
        csv_istr = csv.reader(istr)
        for row in csv_istr :
            yield row[0], row[3]


def data_to_file(header, data, path):
    with open(path, "w") as ostr :
        csv_ostr = csv.writer(ostr)
        csv_ostr.writerow(header)
        for record in data :
            csv_ostr.writerow(record)


def data_from_file(path, with_header=True):
    with open(path, "r") as istr:
        if with_header: istr.readline()
        csv_istr = csv.reader(istr)
        return zip(*(row for row in csv_istr))


def to_chunks(iterable, chunk_size=BATCH_SIZE):
    i = iter(iterable)
    chunk = list(islice(i, chunk_size))
    while chunk:
        yield chunk
        chunk = list(islice(i, chunk_size))
