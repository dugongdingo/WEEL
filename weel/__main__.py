import argparse
import collections
import csv
import datetime
import os
import pickle
import random
import shutil

from .nlgpipeline.preprocess import from_file, Vocab, FastTextVocab, EOS, SOS
from .nlgpipeline.network import Seq2SeqModel

def print_now(line) :
    print(datetime.datetime.now(), ":", line)

from .settings import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0)
parser.add_argument("-l", "--learningrate", type=float, default=0)
parser.add_argument("-e", "--epochs", type=int, default=0)
args = parser.parse_args()

DROPOUT = args.dropout or DROPOUT

LEARNING_RATE = args.learningrate or LEARNING_RATE

EPOCHS = args.epochs or EPOCHS

if USE_WIKI :
    from .datareader.wiki_reader import export
else:
    from .datareader.wordnet_reader import export


extraction_prefix = "wiki_" if USE_WIKI else "wn_"
extraction_prefix += "unambig_" if NO_AMBIG else "ambig_"
extraction_prefix += "withexamples_" if KEEP_EXAMPLES else "noexamples_"

extraction_path = os.path.join(DATA_STORAGE, extraction_prefix + "englishentries.csv")

model_path = os.path.join(MODELS_STORAGE, extraction_prefix + "weelmodel.nlgpipeline_fasttext.pickle")

param_prefix = "lr" + str(LEARNING_RATE) +\
    "_d" + str(DROPOUT) +\
    "_e" + str(EPOCHS) +\
    "_"

test_result_path = os.path.join(
    RESULTS_STORAGE,
    extraction_prefix + param_prefix + "weel.nlgpipeline_fasttext.results.csv"
)

# DATA
if USE_WIKI :
    if not os.path.isdir(DATA_STORAGE) :
        try :
            os.makedirs(DATA_STORAGE)
            path_to_wiki = sys.argv[1]
            shutils.copyfile(path_to_wiki, os.path.join(DATA_STORAGE, os.path.basename(path_to_wiki)))
            print_now("retrieving data...")
            export(path_to_wiki, extraction_path)
        except KeyError :
            print("I need the path towards a wiktionary dump to start up.")
else :
    if not os.path.isfile(extraction_path) :
        print_now("retrieving data...")
        export(extraction_path, unambiguous=NO_AMBIG, with_example=KEEP_EXAMPLES)


print_now("selecting data...")
data = list(from_file(extraction_path))
random.shuffle(data)
words, definitions = zip(*data)
proportion = int(7 * len(words) /10)
input_train, output_train = words[:proportion], definitions[:proportion]
input_test, output_test = words[proportion:], definitions[proportion:]

# GENERATION MODEL
make_model = True
model = None
if not os.path.isdir(MODELS_STORAGE) :
    os.makedirs(MODELS_STORAGE)
    make_model = True
make_model = make_model or not os.path.isfile(model_path)

if make_model :
    print_now("building model...")
    enc_voc = FastTextVocab(PATH_TO_FASTTEXT)
    enc_voc.encrypt_all(input_train + input_test, compute=True)
    input_train = enc_voc.encrypt_all(input_train)
    dec_voc, output_train = Vocab.process(output_train, preprocess=lambda seq:[SOS] + seq.split() + [EOS])
    max_length = max(max(map(len, input_train)), max(map(len, output_train)))
    model = Seq2SeqModel(
        256,
        len(dec_voc),
        enc_voc,
        dec_voc,
        max_length=max_length,
        dropout_p=DROPOUT,
        learning_rate=LEARNING_RATE,
    )
    print_now("training model...")
    model.train(input_train, output_train, epochs=EPOCHS)

    print_now("saving model...")
    with open(model_path, "wb") as ostr :
        pickle.dump(model, ostr)
else :
    print_now("loading model...")
    with open(model_path, "rb") as istr:
        model = pickle.loads(istr.read())

# TESTING
print_now("testing model...")
with open(test_result_path, "w") as ostr:
    csv_writer = csv.writer(ostr)
    csv_writer.writerow(["Word", "Definition", "Prediction"])
    input_test_encrypted = model.encoder_vocab.encrypt_all(input_test)
    predictions = model.decoder_vocab.decrypt_all(map(model.run, input_test_encrypted))
    for word, prediction, definition in zip(input_test, predictions, output_test) :
        csv_writer.writerow([word, definition, prediction])

print_now("all done!")
