import collections
import csv
import itertools

#TODO: mutliple languages in single file support?

def from_file(datafile) :
    """
    create (word, definition) pairs from parsed enwiktionary/ WordNet
    """
    with open(datafile, "r") as istr :
        istr.readline() #skip header
        csv_istr = csv.reader(istr)
        for row in csv_istr :
            yield row[0], row[3]

SOS = "<SOS>"
EOS = "<EOS>"

class Vocab :
    """
    class util for text processing
    """
    def __init__(self) :
        self._index = itertools.count()
        self._vocab = collections.defaultdict(lambda:next(self._index))
        self._counter = collections.Counter()
        added_by_default = self._vocab[SOS], self._vocab[EOS]

    def add(self, word) :
        self.Counter.add(word)
        return self._vocab[word]

    def __getitem__(self, word) :
        return self._vocab[word]

    def __delitem__(self, word):
        del self._counter[word]
        del self._vocab[word]

    @property
    def vocab2index(self) :
        return dict(self._vocab)

    @property
    def index2vocab(self) :
        return {v:k for k,v in self._vocab}

    def __len__(self) :
        return len(self._vocab)

    def __contains__(self, item) :
        return item in self._vocab

    def __iter__(self) :
        return iter(self._vocab)

    def token_count(self) :
        return self.__len__()

    def type_count(self) :
        return sum(v for k, v in self._counter)

def process(sequences, preprocess=None) :
    """
    Builds a Vocab for all words sequences
    Returns the Vocab object and the processed sequences
    """
    if preprocess : sequences = map(preprocess, sequences)
    voc = Vocab()
    return voc, [[voc[w] for w in seq] for seq in sequences]

if __name__ == "__main__" :
    v, seqs  = process(from_file("../data/wiki_english_entries.csv"), preprocess=lambda t:t[0])
    print(len(v))