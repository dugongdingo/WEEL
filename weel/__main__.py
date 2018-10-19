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

#TODO: transfer settings & prefixes computation to settings.py

def print_now(line) :
    """
    utility function: prepend timestamp to std output
    """
    print(datetime.datetime.now(), ":", line)

from .settings import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0)
parser.add_argument("-l", "--learningrate", type=float, default=0)
parser.add_argument("-e", "--epochs", type=int, default=0)
parser.add_argument("-r", "--retrain", action="store_true", default=None)
args = parser.parse_args()

DROPOUT = args.dropout or DROPOUT

LEARNING_RATE = args.learningrate or LEARNING_RATE

EPOCHS = args.epochs or EPOCHS

RETRAIN = args.retrain or RETRAIN

if USE_WIKI :
    from .datareader.wiki_reader import export
else:
    from .datareader.wordnet_reader import export


extraction_prefix = "wiki_" if USE_WIKI else "wn_"
extraction_prefix += "unambig_" if NO_AMBIG else "ambig_"
extraction_prefix += "withexamples_" if KEEP_EXAMPLES else "noexamples_"
extraction_prefix += "nomwe_" if NO_MWE else "withmwe_"

extraction_path = os.path.join(DATA_STORAGE, extraction_prefix + "englishentries.csv")

model_path = os.path.join(MODELS_STORAGE, extraction_prefix + "weelmodel.nlgpipeline_fasttext.pickle")

param_prefix = "lr" + str(LEARNING_RATE) +\
    "_d" + str(DROPOUT) +\
    "_e" + str(EPOCHS) +\
    "_"

if RETRAIN :
    param_prefix += "retrain_"
else:
    param_prefix += "noretrain_"

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
    if not os.path.isdir(extraction_path) :
        os.makedirs(extraction_path)
        print_now("retrieving data...")
        export(extraction_path + ".raw", unambiguous=NO_AMBIG, with_example=KEEP_EXAMPLES, keep_mwe=not NO_MWE)
        print_now("selecting data %s..." % extraction_prefix)
        data = list(from_file(extraction_path + ".raw"))
        random.shuffle(data)
        words, definitions = zip(*data)
        proportion_train = int(8 * len(words) /10)
        proportion_dev = int(9* len(words) /10)
        input_train, output_train = words[:proportion_train], definitions[:proportion_train]
        input_dev, output_dev = words[proportion_train:proportion_dev], definitions[proportion_train:proportion_dev]
        input_test, output_test = words[proportion_dev:], definitions[proportion_dev:]
        def data_to_file(input_data, output_data, path):
            with open(path, "w") as ostr :
                csv_ostr = csv.writer(ostr)
                csv_ostr.writerow(["input", "output"])
                for record in zip(input_data, output_data) :
                    csv_ostr.writerow(record)
        data_to_file(input_test, output_test, os.path.join(extraction_path, "test.csv"))
        data_to_file(input_dev, output_dev, os.path.join(extraction_path, "dev.csv"))
        data_to_file(input_train, output_train, os.path.join(extraction_path, "train.csv"))
    else:
        print_now("selecting data %s..." % extraction_prefix)
        def data_from_file(path):
            with open(path, "r") as istr:
                istr.readline()
                csv_istr = csv.reader(istr)
                return zip(*(row for row in csv_istr))
        input_test, output_test = data_from_file(os.path.join(extraction_path, "test.csv"))
        input_dev, output_dev = data_from_file(os.path.join(extraction_path, "dev.csv"))
        input_train, output_train = data_from_file(os.path.join(extraction_path, "train.csv"))

if USE_DEV :
    input_test = input_dev
    output_test = output_dev
    
# MODEL GENERATION
make_model = True
model = None
if not os.path.isdir(MODELS_STORAGE) :
    os.makedirs(MODELS_STORAGE)
    make_model = True
make_model = make_model or not os.path.isfile(model_path)

if make_model :
    print_now("building model %s..." % param_prefix)
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
        retrain=RETRAIN,
    )
    print_now("training model %s..." % param_prefix)
    for epoch in map(str, range(1, EPOCHS + 1)):
        data = list(zip(input_train, output_train))
        random.shuffle(data)
        input_train, output_train = zip(*data)
        model.train(input_train, output_train, epoch_number=epoch)

        param_prefix = "lr" + str(LEARNING_RATE) +\
            "_d" + str(DROPOUT) +\
            "_e" + epoch +\
            "_"
        if RETRAIN :
            param_prefix += "retrain_"
        else:
            param_prefix += "noretrain_"
        test_result_path = os.path.join(
            RESULTS_STORAGE,
            extraction_prefix + param_prefix + "weel.nlgpipeline_fasttext.results.csv"
        )
        print_now("testing model...")
        with open(test_result_path, "w") as ostr:
            csv_writer = csv.writer(ostr)
            csv_writer.writerow(["Word", "Definition", "Prediction"])
            input_test_encrypted = model.encoder_vocab.encrypt_all(input_test)
            predictions = model.decoder_vocab.decrypt_all(map(model.run, input_test_encrypted))
            for word, prediction, definition in zip(input_test, predictions, output_test) :
                csv_writer.writerow([word, definition, prediction])


    print_now("saving model...")
    with open(model_path, "wb") as ostr :
        pickle.dump(model, ostr)
else :
    print_now("loading model...")
    with open(model_path, "rb") as istr:
        model = pickle.loads(istr.read())

print_now("all done!")
