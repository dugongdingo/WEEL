#!/usr/bin/env bash

python3 -m venv .venv/weel

source .venv/weel/bin/activate

pip3 install -r requirements.txt

mkdir data
mkdir results
mkdir models

mkdir .fasttext; cd .fasttext
git clone https://github.com/facebookresearch/fastText.git
cd FastText
pip3 install .

cd ../..

wget "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip" -P data
cd data
unzip wiki-news-300d-1M-subword.vec.zip

python3 -c "import nltk; nltk.download('wordnet')"
