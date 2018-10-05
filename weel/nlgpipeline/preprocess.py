import collections
import csv
import itertools

#TODO: mutliple languages in single file support?

def from_file(datafile) :
    """
    create (word, definition) pairs from parsed wiktionary/ WordNet
    """
    with open(datafile, "r") as istr :
        istr.readline() #skip header
        csv_istr = csv.reader(istr)
        for row in csv_istr :
            yield row[0], row[3]

# start of sentence & end of sentence tokens
SOS = "<SOS>"
EOS = "<EOS>"

OOV = "<OOV>"

class Vocab :
    """
    class util for text processing
    """
    def __init__(self) :
        self._index = itertools.count()
        self._vocab = collections.defaultdict(self._next_idx)
        self._counter = collections.Counter()
        added_by_default = self._vocab[SOS], self._vocab[EOS], self._vocab[OOV]

    def _next_idx(self):
        return next(self._index)

    def add(self, word) :
        self._counter.update([word])
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
        return {v:k for k,v in self._vocab.items()}

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

    def decrypt(self, sequence):
        i2v = self.index2vocab
        return [i2v[t] for t in sequence]

    def encrypt(self, sequence, frozen=False):
        if frozen :
            return [self[t] if t in self else self[OOV] for t in sequence]
        return [self.add(t) for t in sequence]

    @classmethod
    def process(cls, sequences, preprocess=None) :
        """
        Builds a Vocab for all words sequences
        Returns the Vocab object and the processed sequences
        """
        if preprocess : sequences = map(preprocess, sequences)
        voc = cls()
        return voc, [voc.encrypt(seq) for seq in sequences]
