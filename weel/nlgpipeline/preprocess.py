import collections
import csv
import itertools

import numpy

import fastText

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
        indexed_by_default = self._vocab[SOS], self._vocab[EOS], self._vocab[OOV]

    def _next_idx(self):
        return next(self._index)

    def trim(self, threshold=5) :
        """
        Remove all words below threshold
        """
        OOV_idx = self[OOV]
        for word in self._vocab :
            if self._counter[word] < threshold :
                self._vocab[word] = OOV_idx

    def add(self, word) :
        """
        Add word, update counts
        """
        self._counter.update([word])
        return self._vocab[word]

    def __getitem__(self, word) :
        return self._vocab[word]

    def __delitem__(self, word):
        del self._counter[word]
        del self._vocab[word]

    @property
    def vocab2index(self) :
        """
        get {word:index} dict
        """
        return dict(self._vocab)

    @property
    def index2vocab(self) :
        """
        get {index:word} dict
        """
        return {v:k for k,v in self._vocab.items()}

    def __len__(self) :
        return len(self._vocab)

    def __contains__(self, item) :
        return item in self._vocab

    def __iter__(self) :
        return iter(self._vocab)

    def type_count(self) :
        """
        Number of distinct items
        """
        return self.__len__()

    def token_count(self) :
        """
        Number of counted items
        """
        return sum(v for k, v in self._counter)

    def decrypt(self, sequence) :
        """
        transform a sequence of indices into a sequence of words
        """
        i2v = self.index2vocab
        return [i2v[t] for t in sequence]

    def decrypt_all(self, sequences) :
        """
        transform sequences of indices into sequences of words
        """
        i2v = self.index2vocab
        return [[i2v[t] for t in seq] for seq in sequences]

    def encrypt(self, sequence, frozen=False) :
        """
        transform a sequence of words into a sequence of indices
        if not frozen, vocabulary counts will be updated
        """
        if frozen :
            return [self[t] if t in self else self[OOV] for t in sequence]
        return [self.add(t) for t in sequence]

    def encrypt_all(self, sequences, frozen=False) :
        """
        transform sequences of words into sequences of indices
        if not frozen, vocabulary counts will be updated
        """
        if frozen :
            return [[self[t] if t in self else self[OOV] for t in seq] for seq in sequences]
        return [[self.add(t) for t in seq] for seq in sequences]

    @classmethod
    def process(cls, sequences, preprocess=None) :
        """
        Builds a Vocab for all words sequences
        Returns the Vocab object and the processed sequences
        """
        if preprocess : sequences = map(preprocess, sequences)
        voc = cls()
        return voc, [voc.encrypt(seq) for seq in sequences]

class FastTextSubWordVocab :
    """
    Embedding lookup utility class based on a FastText model
    """
    def __init__(self, ft_modelpath):
        self.model = fastText.load_model(ft_modelpath)
        self.subseq_dict = {}
        self.lookup = {}
        self.embedding_matrix = None

    def encrypt(self, sequence):
        """
        transform a sequence of words into a sequence of indices
        if not frozen, vocabulary counts will be updated
        """
        return [self.lookup[s] for s in self.subseq_dict[sequence]]

    def encrypt_all(self, sequences, compute=False):
        """
        transform sequences of words into sequences of indices
        if compute, the lookup table and the embedding matrix for the model are computed
        """
        if compute : #TODO: separate computation from encryption in two distinct functions
            _vecs = {}
            for sequence in sequences :
                subwords, indices = self.model.get_subwords(sequence)
                self.subseq_dict[sequence] = subwords
                for subword, index in zip(subwords,indices) :
                    if subword not in _vecs :
                        _vecs[subword] = self.model.get_input_vector(index)
            all_subs = {s for w in self.subseq_dict for s in self.subseq_dict[w]}
            self.embedding_matrix = numpy.zeros((len(all_subs),self.model.get_dimension()))
            del self.model
            for i, s in enumerate(all_subs) :
                self.embedding_matrix[i,:] = _vecs[s]
                del _vecs[s]
                self.lookup[s] = i
        return [self.encrypt(seq) for seq in sequences]

class FastTextVocab :
    """
    Embedding lookup utility class based on a FastText model
    """
    def __init__(self, ft_modelpath):
        self.model = fastText.load_model(ft_modelpath)
        self.lookup = {}
        self.embedding_matrix = None

    def encrypt(self, sequence):
        """
        transform a sequence of words into a sequence of indices
        if not frozen, vocabulary counts will be updated
        """
        return [self.lookup[s] for s in sequence]

    def encrypt_all(self, sequences, compute=False):
        """
        transform sequences of words into sequences of indices
        if compute, the lookup table and the embedding matrix for the model are computed
        """
        if compute : #TODO: separate computation from encryption in two distinct functions
            _vecs = {
                w : self.model.get_word_vector(w)
                for s in sequences
                for w in s
            }
            if not EOS in _vecs:
                _vecs[EOS] = numpy.random.rand(self.model.get_dimension())
            if not SOS in _vecs:
                _vecs[SOS] = numpy.random.rand(self.model.get_dimension())

            self.embedding_matrix = numpy.zeros((len(_vecs),self.model.get_dimension()))
            del self.model
            for i, s in enumerate(_vecs) :
                self.embedding_matrix[i,:] = _vecs[s]
                self.lookup[s] = i
        return [self.encrypt(seq) for seq in sequences]

    def __len__(self):
        return self.embedding_matrix.shape[0]
