"""
file listing all settings:
    - storage paths
    - data parameters (wiki vs. wordnet, ambiguous entries, keeping examples)
    - neural model parameters (learning rate, dropout, nnumber of epochs)
"""
import argparse
import os

import torch

USE_WIKI = False

DATA_STORAGE = "./weel/data"

PATH_TO_WIKI = os.path.join(DATA_STORAGE, "enwiktionary-20181001-pages-meta-current.xml")

PATH_TO_FASTTEXT = os.path.join(DATA_STORAGE, "crawl-300d-2M-subword/crawl-300d-2M-subword.bin")

MODELS_STORAGE = "./weel/models"

RESULTS_STORAGE = "./weel/results"

NO_AMBIG = True

NO_MWE = True

KEEP_EXAMPLES = False

LEARNING_RATE = 0.001

DROPOUT = 0.01

EPOCHS = 10

RETRAIN = False

USE_DEV = True

HIDDEN_SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0.)
parser.add_argument("-l", "--learningrate", type=float, default=0.)
parser.add_argument("-e", "--epochs", type=int, default=0.)
parser.add_argument("-r", "--retrain", action="store_true", default=None)
args = parser.parse_args()

DROPOUT = args.dropout or DROPOUT

LEARNING_RATE = args.learningrate or LEARNING_RATE

EPOCHS = args.epochs or EPOCHS

RETRAIN = args.retrain or RETRAIN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100

MAKE_MODEL = True

N_LAYERS = 1

CLIP = 50.

BATCH_SIZE = 150
