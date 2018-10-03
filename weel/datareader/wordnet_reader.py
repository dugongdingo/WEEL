import csv

from nltk.corpus import wordnet as wn

_POS = [
    'n',
    'v',
    'a',
    'r',
]

def retrieve_definitions(extraction_dictionary):
    for ss in wn.all_synsets() :
        for l in extraction_dictionary :
            for p in extraction_dictionary[l] :
                if ss.pos() == p :
                    for lemma in ss.lemmas(lang=l):
                        for ex in ss.examples() :
                            if lemma.name() in set(ex.split()) :
                                yield lemma.name(), l, p, ss.definition(), ex

if __name__ == "__main__" :
    extraction_dictionary = {'eng': _POS}
    with open("../data/wn_english_entries.csv", "w") as ostr:
        csv_ostr = csv.writer(ostr)
        csv_ostr.writerow(["title", "language", "POS", "definition", "example"])
        for entry in sorted(set(retrieve_definitions(extraction_dictionary))):
            csv_ostr.writerow(entry)
