import bs4
import re

_LANG_PATTERNS = {
    'EN' : re.compile("^==English==$", re.MULTILINE),
}

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
                        yield title, lang, pos, pos_section

if __name__ == "__main__" :
    filepath = "../../Desktop/enwiktionary-20180901-pages-meta-current.xml"
    extraction_dictionary = {'EN': ['Noun']}
    print(next(retrieve_definitions(filepath, extraction_dictionary)))
