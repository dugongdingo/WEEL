import collections
import os
import shutil

from .datareader.wordnet_reader import export
from .nlgpipeline.preprocess import from_file, Vocab, EOS, SOS
from .nlgpipeline.network import Seq2SeqModel

DATA_STORAGE = "./weel/data"

extraction_path = os.path.join(DATA_STORAGE, "wn_english_entries.csv")

"""if not os.path.isdir(DATA_STORAGE) :
    path_to_wiki = sys.argv[1]
    os.makedirs(DATA_STORAGE)
    shutils.copyfile(path_to_wiki, os.path.join(DATA_STORAGE, os.path.basename(path_to_wiki)))
"""
print("retrieving data...")
export(extraction_path)
input, output = zip(*sorted(set(from_file(extraction_path))))

print("selecting unambiguous data...")
defs = collections.Counter(input)
input, output = [i for i in input if defs[i] == 1], [o for i,o in zip(input, output) if defs[i] == 1]

print("dumping data...")
with open("weel/data/wn_unambiguous_words.csv", "w") as ostr:
    for i, o in zip(input, output) :
        print(i, o, sep="\t", file=ostr)

print("loading model...")
enc_voc, input = Vocab.process(input, preprocess=lambda seq: list(seq) + [EOS])
dec_voc, output = Vocab.process(output, preprocess=lambda seq:[SOS] + seq.split() + [EOS])
model = Seq2SeqModel(len(enc_voc), 256, len(dec_voc), enc_voc, dec_voc)

print("training model...")
model.train(input, output, len(input))
with open("model.pickle", "wb") :
    pickle.dump(model, ostr)
words, _ = model.run(input[0])
print(input[0], ":", words)
