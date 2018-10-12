# WEEL

WEEL - Word Embeddings Experiments with Linguality

python3 project: _generating definitions using monolingual and/or multilingual embeddings_

install dependencies using `pip3 -r requirements.txt`.

nltk also requires you install wordnet: type in python3 shell :
`import nltk
nltk.download('wordnet')`

The fasttext pypi package cannot handle the models downloadable via Facebook Research.
Try installing fastText (mind the case) : see https://github.com/facebookresearch/fastText/tree/master/python

The necessary FastText model is available on the dedicated web page: https://fasttext.cc/docs/en/english-vectors.html

run with `python3 -m weel`.
