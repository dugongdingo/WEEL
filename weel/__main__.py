import collections
import csv
import datetime
import nltk.tokenize
import os
import pickle
import random
import shutil

from .utils import print_now, read_parsed_data_file, data_to_file, data_from_file
from .nlgpipeline.preprocess import EOS, SOS, compute_lookup, translate, reverse_lookup
from .nlgpipeline.network import Seq2SeqModel

#TODO: transfer settings & prefixes computation to settings.py

from .settings import *

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
        data = list(read_parsed_data_file(extraction_path + ".raw"))
        random.shuffle(data)
        words, definitions = zip(*data)
        proportion_train = int(8 * len(words) /10)
        proportion_dev = int(9* len(words) /10)
        input_train, output_train = words[:proportion_train], definitions[:proportion_train]
        input_dev, output_dev = words[proportion_train:proportion_dev], definitions[proportion_train:proportion_dev]
        input_test, output_test = words[proportion_dev:], definitions[proportion_dev:]

        header = ["input", "output"]
        data_to_file(
            header,
            zip(input_test, output_test),
            os.path.join(extraction_path, "test.csv")
        )
        data_to_file(
            header,
            zip(input_dev, output_dev),
            os.path.join(extraction_path, "dev.csv")
        )
        data_to_file(
            header,
            zip(input_train, output_train),
            os.path.join(extraction_path, "train.csv")
        )
    else:
        print_now("selecting data %s..." % extraction_prefix)

        input_test, output_test = data_from_file(os.path.join(extraction_path, "test.csv"))
        input_dev, output_dev = data_from_file(os.path.join(extraction_path, "dev.csv"))
        input_train, output_train = data_from_file(os.path.join(extraction_path, "train.csv"))

input_rest = []
output_rest = []

if USE_DEV :
    input_rest = input_test
    output_rest = output_test
    input_test = input_dev
    output_test = output_dev
else :
    input_rest = input_dev
    output_rest = output_dev

# MODEL GENERATION
model = None
if not os.path.isdir(MODELS_STORAGE) :
    os.makedirs(MODELS_STORAGE)
    MAKE_MODEL = True
MAKE_MODEL = MAKE_MODEL or not os.path.isfile(model_path)

if MAKE_MODEL :
    print_now("building model %s..." % param_prefix)
    input_all = input_train + input_test + input_rest
    subword_lookup, subword_embeddings = compute_lookup(input_all, PATH_TO_FASTTEXT, use_subwords=True)
    input_train = translate(input_train, subword_lookup, use_subwords=True)
    output_all = list(map(nltk.tokenize.word_tokenize, output_train + output_test + output_rest))
    decoder_lookup, decoder_embeddings = compute_lookup(output_all, PATH_TO_FASTTEXT)
    output_train = translate(
        [[SOS] + nltk.tokenize.word_tokenize(o) + [EOS] for o in output_train],
        decoder_lookup,
    )

    max_length = max(max(map(len, input_train)), max(map(len, output_train)))
    model = Seq2SeqModel(
        256,
        len(decoder_lookup),
        subword_embeddings,
        decoder_embeddings,
        decoder_lookup[SOS],
        decoder_lookup[EOS],
        max_length=max_length,
        dropout_p=DROPOUT,
        learning_rate=LEARNING_RATE,
        retrain=RETRAIN,
    )
    print_now("training model %s..." % param_prefix)
    for epoch in map(str, range(1, EPOCHS + 1)):
        data = list(zip(input_train, output_train))
        random.shuffle(data)
        input_train, output_train = zip(*data[:10])
        train_losses = model.train(input_train, output_train, epoch_number=epoch)

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
        test_losses = []
        with open(test_result_path, "w") as ostr:
            csv_writer = csv.writer(ostr)
            csv_writer.writerow(["Word", "Definition", "Prediction"])
            input_test_encrypted = translate(input_test, subword_lookup, use_subwords=True)
            output_test_encrypted = translate(
                [[SOS] + nltk.tokenize.word_tokenize(o) + [EOS] for o in output_test],
                decoder_lookup,
            )
            predictions, test_losses = zip(*(model.run(i, o) for i, o in zip(input_test_encrypted, output_test_encrypted)))
            predictions = translate(predictions, reverse_lookup(decoder_lookup))
            for word, prediction, definition in zip(input_test, predictions, output_test) :
                csv_writer.writerow([word, definition, prediction])
        print_now(
            "avg loss train:",
            sum(train_losses)/len(train_losses),
            ", avg loss test:",
            sum(test_losses)/len(test_losses)
        )


    print_now("saving model...")
    with open(model_path, "wb") as ostr :
        pickle.dump(model, ostr)
else :
    print_now("loading model...")
    with open(model_path, "rb") as istr:
        model = pickle.loads(istr.read())

print_now("all done!")
