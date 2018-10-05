import collections
import csv
import datetime
import os
import pickle
import shutil

from .datareader.wordnet_reader import export
from .nlgpipeline.preprocess import from_file, Vocab, EOS, SOS
from .nlgpipeline.network import Seq2SeqModel

def print_now(line) :
    print(datetime.datetime.now(), ":", line)

DATA_STORAGE = "./weel/data"

MODELS_STORAGE = "./weel/models"

extraction_path = os.path.join(DATA_STORAGE, "wn_english_entries.csv")

model_path = os.path.join(MODELS_STORAGE, "weel.nlg_pipeline.pickle")

test_result_path = os.path.join(DATA_STORAGE, "weel.nlg_pipeline.results.csv")

# DATA
if not os.path.isdir(DATA_STORAGE) :
    path_to_wiki = sys.argv[1]
    os.makedirs(DATA_STORAGE)
    shutils.copyfile(path_to_wiki, os.path.join(DATA_STORAGE, os.path.basename(path_to_wiki)))
    print_now("retrieving data...")
    export(extraction_path)

input, output = zip(*sorted(set(from_file(extraction_path))))

print_now("selecting unambiguous data...")
defs = collections.Counter(input)
words, definitions = [i for i in input if defs[i] == 1], [o for i,o in zip(input, output) if defs[i] == 1]
proportion = int(7 * len(words) /10)
input_train, output_train = words[:proportion], definitions[:proportion]
input_test, output_test = words[proportion:], definitions[proportion:]
print_now("dumping data...")
with open("weel/data/wn_unambiguous_words.csv", "w") as ostr:
    for i, o in zip(words, defs) :
        print(i, o, sep="\t", file=ostr)

# GENERATION MODEL
make_model = False
model = None
if not os.path.isdir(MODELS_STORAGE) :
    os.makedirs(MODELS_STORAGE)
    make_model = True
make_model = make_model or not os.path.isfile(model_path)

if make_model :
    print_now("building model...")
    enc_voc, input_train = Vocab.process(input_train, preprocess=lambda seq: list(seq) + [EOS])
    dec_voc, output_train = Vocab.process(output_train, preprocess=lambda seq:[SOS] + seq.split() + [EOS])
    model = Seq2SeqModel(len(enc_voc), 256, len(dec_voc), enc_voc, dec_voc)

    print_now("training model...")
    model.train(input_train, output_train, len(input_train))

    print_now("saving model...")
    with open(model_path, "wb") as ostr :
        pickle.dump(model, ostr)
else :
    print_now("loading model...")
    with open(model_path, "rb") as istr:
        model = pickle.loads(istr)

# TESTING
print_now("testing model...")
with open(test_result_path, "w") as ostr:
    csv_writer = csv.writer(ostr)
    csv_writer.writerow(["Word", "Definition", "Prediction"])
    input_test = [model.encoder_vocab.encrypt(ipt, frozen=True) for ipt in input_test]
    for word, definition in zip(input_test, output_test) :
        prediction = model.run(word)
        csv_writer.writerow([model.encoder_vocab.decrypt(word), definition, prediction])

print_now("all done!")
