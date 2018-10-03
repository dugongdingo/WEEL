import collections
import os
import shutil

from .datareader.wiki_reader import export
from .nlgpipeline.preprocess import from_file, Vocab, EOS, SOS
from .nlgpipeline.network import Seq2SeqModel

DATA_STORAGE = "./weel/data"
extraction_path = os.path.join(DATA_STORAGE, "wiki_english_entries.csv")

if not os.path.isdir(DATA_STORAGE) :
    path_to_wiki = sys.argv[1]
    os.makedirs(DATA_STORAGE)
    shutils.copyfile(path_to_wiki, os.path.join(DATA_STORAGE, os.path.basename(path_to_wiki)))
    export(path_to_wiki, extraction_path)

input, output = zip(*from_file(extraction_path))

defs = collections.Counter(input)
input, output = zip(*[[i,o] for i,o in zip(input, output) if defs[i] == 1])

enc_voc, input = Vocab.process(input, preprocess=lambda seq: list(seq) + [EOS])
dec_voc, output = Vocab.process(output, preprocess=lambda seq:[SOS] + seq.split() + [EOS])

model = Seq2SeqModel(len(enc_voc), 256, len(dec_voc), enc_voc, dec_voc)
model.train(input, output, len(input))
words, _ = model.run(input[0])
print(words)
