import bs4
import re


# TODO: add other language codes
_LANG_PATTERNS = {
    'EN' : re.compile(r"^==English==$", re.MULTILINE),
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

_XML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.MULTILINE | re.DOTALL)

_POS_PATTERNS = {pos : re.compile(r"^===%s===$" % re.escape(pos), re.MULTILINE) for pos in _POS}

_NEXT_L2_SECTION_PATTERN = re.compile(r"^==[^=]+?==$", re.MULTILINE)

_NEXT_L3_SECTION_PATTERN = re.compile(r"^===[^=]+?===$", re.MULTILINE)

_NEXT_L4_SECTION_PATTERN = re.compile(r"^====[^=]+?====$", re.MULTILINE)

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
        # next section closes the relevant subsection
        next_section = _NEXT_L4_SECTION_PATTERN.search(section_text)
        if next_section :
            section_text = section_text[:next_section.start()]
        else :
            next_section = _NEXT_L3_SECTION_PATTERN.search(section_text)
            if next_section :
                section_text = section_text[:next_section.start()]
            else :
                next_section = _NEXT_L2_SECTION_PATTERN.search(section_text)
                if next_section :
                    section_text = section_text[:next_section.start()]
        return section_text
    raise ValueError

def is_definition(line) :
    return _DEFINITION_PATTERN.match(line)

def is_quote(line) :
    return _QUOTE_PATTERN.match(line)

def is_example(line) :
    return _EXAMPLE_PATTERN.match(line)

def remove_XML_comments(section) :
    """
    Removes XML inline comments
    """
    results = list(_XML_COMMENT_PATTERN.finditer(section))
    if not len(results) : return section
    for matched in reversed(results) :
        section = section[:matched.start()] + section[matched.end():]
    return section


def _parse_meta(char_itr) :
    """
    Parse meta ( {{meta}} ) blocks
    """
    meta = ""
    while True :
        c0, c1, c2 = next(char_itr)
        if c0 == '}' and c1 == '}' :
            drop = next(char_itr)
            meta = meta.split("|")
            if len(meta) > 1 :
                if meta[0] == "ux" :
                    return meta[-1]
                try:
                    return dict(t for t in (s.split("=") for s in meta) if len(t) == 2)["passage"]
                except KeyError:
                    pass
            return ""
        elif c0 == '[' and c1 == '[' :
            drop = next(char_itr)
            meta += _parse_link(char_itr)
        elif c0 == '{' and c1 == '{':
            drop = next(char_itr)
            meta += _parse_meta(char_itr)
        elif c0 == '\'' and c1 == '\'':
            if c2 == '\'' :
                drop = next(char_itr)
            drop = next(char_itr)
        else :
            meta += c0 if c0 not in "\n\r" else " "


def _parse_link(char_itr) :
    """
    Parse link ( [[link]] ) blocks
    """
    link = ""
    while True :
        c0, c1, c2 = next(char_itr)
        if c0 == ']' and c1 == ']' :
            drop = next(char_itr)
            return link
        elif c0 == '|' :
            link = ""
        elif c0 == '[' and c1 == '[' :
            drop = next(char_itr)
            link += _parse_link(char_itr)
        elif c0 == '{' and c1 == '{':
            drop = next(char_itr)
            link += _parse_meta(char_itr)
        elif c0 == '\'' and c1 == '\'':
            if c2 == '\'' :
                drop = next(char_itr)
            drop = next(char_itr)
        else :
            link += c0 if c0 not in "\n\r" else " "


def cleanup_wikisyntax(section) :
    """
    Cleans up a section using a character iterator with two characters look-ahead.
    Recursive syntax is dealt with using recursive calls
    """
    char_itr = zip(section, section[1:], section[2:])
    out = ""
    while True :
        try :
            c0, c1, c2 = next(char_itr)
            if c0 == c1 :
                if c0 == '\'' : #drop bold & italic formatting
                    if c0 == c2 :
                        drop = next(char_itr)
                    drop = next(char_itr)
                elif c0 == '[' : # links
                    drop = next(char_itr)
                    out += _parse_link(char_itr)
                elif c0 == '{' : # metas
                    drop = next(char_itr)
                    out += _parse_meta(char_itr)
                else :
                    out += c0
            else :
                out += c0
        except StopIteration :
            break
    return out


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
                        pos_section = remove_XML_comments(pos_section)
                        pos_section = cleanup_wikisyntax(pos_section)
                        definition = ""
                        for line in pos_section.split("\n") :
                            line = line.strip()
                            if is_definition(line) :
                                definition = line
                            elif is_quote(line) :
                                yield title, lang, pos, definition, line, "quote"
                            elif is_example(line) :
                                yield title, lang, pos, definition, line, "example"


def export(filepath, export_path) :
    import csv
    extraction_dictionary = {'EN': ['Noun']}
    with open(export_path, "w") as ostr:
        csv_ostr = csv.writer(ostr)
        csv_ostr.writerow(["title", "language", "POS", "definition", "example", "example type"])
        for entry in retrieve_definitions(filepath, extraction_dictionary) :
            csv_ostr.writerow(entry)


if __name__ == "__main__" :
    import sys
    export(sys.argv[1], "./weel/data/wiki_english_entries.csv")
