import bs4
import re

# TODO: add other language codes
_LANG_PATTERNS = {
    'EN' : re.compile("^==English==$", re.MULTILINE),
}

# From wikipedia
_POS = [
    "Adjective",
    "Adverb",
    "Ambiposition",
    "Article",
    "Circumposition",
    "Classifier",
    "Conjunction",
    "Contraction",
    "Counter",
    "Determiner",
    "Ideophone",
    "Interjection",
    "Noun",
    "Numeral",
    "Participle",
    "Particle",
    "Postposition",
    "Preposition",
    "Pronoun",
    "Proper noun",
    "Verb",
]

_POS_PATTERNS = {pos : re.compile(r"^===%s===$" % re.escape(pos), re.MULTILINE) for pos in _POS}

_NEXT_L2_SECTION_PATTERN = re.compile("^==[^=]+?==$", re.MULTILINE)

_NEXT_L4_SECTION_PATTERN = re.compile("^====[^=]+?====$", re.MULTILINE)

_DEFINITION_PATTERN = re.compile(r"^#+ .*$")

_QUOTE_PATTERN = re.compile(r"^#+[*] .*$")

_EXAMPLE_PATTERN = re.compile(r"^#+: .*$")

def read_pages(filename) :
    """
    Yields pages containing <title>...</title> and <text>...</text>
    from wiki dump given as path
    """
    def _yield_raw(filename):
        with open(filename, "r") as istr:
    	    pg_acc = []
    	    for line in istr :
                if line.strip() == "<page>" :
                    pg_acc = []
                pg_acc.append(line)
                if line.strip() == "</page>" :
                    yield pg_acc
    for page in _yield_raw(filename) :
        page_soup = bs4.BeautifulSoup("".join(page), "lxml")
        title = page_soup.find("title")
        if not title : continue
        title = title.get_text()
        text = page_soup.find("text")
        if not text : continue
        text = text.get_text()
        yield title, text

def page_contains_lang(page_text, lang_code) :
    """
    Checks whether page contains entry for Language
    """
    return _LANG_PATTERNS[lang_code].search(page_text)

def page_contains_pos(page_text, pos_name):
    """
    Checks whether page contains entry for Part of Speech
    """
    return _POS_PATTERNS[pos_name].search(page_text)

def extract_lang_section(page_text, lang_code) :
    """
    Retrieves relevant wiki section for language
    """
    lang_section = page_contains_lang(page_text, lang_code)
    if lang_section :
        section_text = page_text[lang_section.end():]
        next_section = _NEXT_L2_SECTION_PATTERN.search(section_text)
        if next_section : section_text = section_text[:next_section.start()]
        return section_text
    raise ValueError

def extract_pos_section(page_text, pos_name) :
    """
    Retrieves relevant wiki section for PoS
    """
    pos_section = page_contains_pos(page_text, pos_name)
    if pos_section :
        section_text = page_text[pos_section.end():]
        next_section = _NEXT_L4_SECTION_PATTERN.search(section_text)
        if next_section : section_text = section_text[:next_section.start()]
        return section_text
    raise ValueError

def is_definition(line) :
    return _DEFINITION_PATTERN.match(line)

def is_quote(line) :
    return _QUOTE_PATTERN.match(line)

def is_example(line) :
    return _EXAMPLE_PATTERN.match(line)

def retrieve_definitions(filepath, extraction_dictionary) :
    """
    Retrieves definitions as specified by `extraction_dictionary`
    """
    for title, text in read_pages(filepath) :
        for lang in extraction_dictionary :
            if page_contains_lang(text, lang) :
                lang_section = extract_lang_section(text, lang)
                for pos in extraction_dictionary[lang] :
                    if page_contains_pos(lang_section, pos) :
                        pos_section = extract_pos_section(lang_section, pos)
                        definition = ""
                        for line in pos_section.split("\n") :
                            line = line.strip()
                            if is_definition(line) :
                                definition = line
                            elif is_quote(line) :
                                yield title, lang, pos, definition, line, "quote"
                            elif is_example(line) :
                                yield title, lang, pos, definition, line, "example"


if __name__ == "__main__" :
    import csv
    filepath = "../data/enwiktionary-20180901-pages-meta-current.xml"
    extraction_dictionary = {'EN': _POS}
    with open("../data/wiki_english_entries.csv", "w") as ostr:
        csv_ostr = csv.writer(ostr)
        csv_ostr.writerow(["title", "language", "POS", "definition", "example", "example type"])
        for entry in retrieve_definitions(filepath, extraction_dictionary) :
            csv_ostr.writerow(entry)
