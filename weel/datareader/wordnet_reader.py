import csv

from nltk.corpus import wordnet as wn

_POS = [
    'n',
    'v',
    'a',
    'r',
    's',
]


def retrieve_definitions(extraction_dictionary, with_example=True):
    """
    read definitions from nltk wordnet
    """
    for ss in wn.all_synsets() :
        for l in extraction_dictionary :
            for p in extraction_dictionary[l] :
                if ss.pos() == p :
                    for lemma in ss.lemmas(lang=l):
                        if with_example :
                            for ex in ss.examples() :
                                if lemma.name().lower() in set(ex.split()) :
                                    yield lemma.name().lower(), l, p, ss.definition(), ex
                        else :
                            yield lemma.name().lower(), l, p, ss.definition()

def export(export_path, unambiguous=False, with_example=False, keep_mwe=True) :
    """
    parse data and write it to a csv file
    """
    retrieving_func = retrieve_unambiguous if unambiguous else retrieve_definitions
    extraction_dictionary = {'eng': ['n']}
    with open(export_path, "w") as ostr:
        csv_ostr = csv.writer(ostr)
        header = ["title", "language", "POS", "definition"]
        if with_example :
            header += ["example"]
        csv_ostr.writerow(header)
        for entry in sorted(set(retrieving_func(extraction_dictionary, with_example=with_example, keep_mwe=keep_mwe))) :
            csv_ostr.writerow(entry)

def retrieve_unambiguous(extraction_dictionary, with_example=True, keep_mwe=True) :
    """
    read unambiguous definitions from nltk wordnet
    """
    lemma_data = set()
    if keep_mwe :
        lemma_data = {(lm.lower(), lg)
            for lg in extraction_dictionary
            for p in extraction_dictionary[lg]
            for lm in wn.all_lemma_names(lang=lg, pos=p)
        }
    else :
        lemma_data = {(lm.lower(), lg)
            for lg in extraction_dictionary
            for p in extraction_dictionary[lg]
            for lm in wn.all_lemma_names(lang=lg, pos=p)
            if "_" not in lm
        }
    for lemma, lang in lemma_data :
        synsets = wn.synsets(lemma)
        if len(synsets) == 1 :
            synset = synsets[0]
            if with_example :
                examples = {ex for ex in synset.examples() if lemma_name in set(ex.split())}
                if not len(examples) :
                    yield lemma, lang, synset.pos(), synset.definition(), ""
                else :
                    for example in examples :
                        yield lemma, lang, synset.pos(), synset.definition(), example
            else :
                yield lemma, lang, synset.pos(), synset.definition()
