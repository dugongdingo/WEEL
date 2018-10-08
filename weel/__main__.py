import collections
import csv
import datetime
import os
import pickle
import shutil

from .nlgpipeline.preprocess import from_file, Vocab, EOS, SOS
from .nlgpipeline.network import Seq2SeqModel

def print_now(line) :
    print(datetime.datetime.now(), ":", line)

USE_WIKI = False

if USE_WIKI :
    from .datareader.wiki_reader import export
else:
    from .datareader.wordnet_reader import export

DATA_STORAGE = "./weel/data"

MODELS_STORAGE = "./weel/models"

NO_AMBIG = True

KEEP_EXAMPLES = False

extraction_prefix = "wiki_" if USE_WIKI else "wn_"
extraction_prefix += "unambig_" if NO_AMBIG else "ambig_"
extraction_prefix += "withexamples_" if KEEP_EXAMPLES else "noexamples_"

extraction_path = os.path.join(DATA_STORAGE, extraction_prefix + "englishentries.csv")

model_path = os.path.join(MODELS_STORAGE, extraction_prefix + "weelmodel.nlgpipeline.pickle")

test_result_path = os.path.join(DATA_STORAGE, extraction_prefix + "weel.nlgpipeline.results.csv")

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
words, definitions = zip(*from_file(extraction_path))
proportion = int(7 * len(words) /10)
input_train, output_train = words[:proportion], definitions[:proportion]
input_test, output_test = words[proportion:], definitions[proportion:]

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
    max_length = max(max(map(len, input_train)), max(map(len, output_train)))
    model = Seq2SeqModel(len(enc_voc), 256, len(dec_voc), enc_voc, dec_voc, max_length=max_length)

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
