import os

USE_WIKI = False

DATA_STORAGE = "./weel/data"

MODELS_STORAGE = "./weel/models"

RESULTS_STORAGE = "./weel/results"

NO_AMBIG = True

KEEP_EXAMPLES = False

PATH_TO_FASTTEXT = os.path.join(DATA_STORAGE, "crawl-300d-2M-subword/crawl-300d-2M-subword.bin")

LEARNING_RATE = 0.001

DROPOUT = 0.1

EPOCHS = 5
