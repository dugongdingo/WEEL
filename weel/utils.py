import csv
import datetime

import numpy
import nltk.tokenize
import torch

from .settings import DEVICE

SOS = "<SOS>"

EOS = "<EOS>"

OOV = "<OOV>"

def to_tensor(seq, device=DEVICE) :
    return torch.tensor(seq, dtype=torch.long, device=device).view(-1, 1)


def to_sentence(raw_str):
    return [SOS] + nltk.tokenize.word_tokenize(raw_str) + [EOS]


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
