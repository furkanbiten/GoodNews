import json
import nltk
import spacy
import numpy as np
import tqdm
import unidecode
from bs4 import BeautifulSoup
import re
import unicodedata
from itertools import groupby

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

# def stem_words(words):
#     """Stem words in list of tokenized words"""
#     stemmer = LancasterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems
#
# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas

def normalize(words):
    words = remove_non_ascii(words)
#     words = to_lowercase(words)
    words = remove_punctuation(words)
#     words = replace_numbers(words)
#     words = remove_stopwords(words)
    return words

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def preprocess_sentence(sen):
    sen = sen.strip()
#     sen = re.sub(uri_re, "", sen)
    sen = sen.encode('ascii',errors='ignore')
    sen = unidecode.unidecode(sen)
    sen = denoise_text(sen)
    # sen = replace_contractions(sen)
    sen = nltk.tokenize.word_tokenize(sen)
    sen = normalize([unicode(s) for s in sen])
#     sen = normalize(unicode(sen))
#     return sen
    return sen
#     tokenized = nltk.tokenize.word_tokenize(temp)
#     final = normalize(unicode(tokenized))
#     return ''.join(final)

# def NER(sen):
#     doc = nlp(unicode(sen))
#     return [d.ent_iob_+'-'+d.ent_type_ if d.ent_iob_ != 'O' else d.text for d in doc ], [d.text for d in doc]
def NER(sen):
    doc = nlp(unicode(sen))
#     text = doc.text
#     for ent in doc.ents:
#         text = text.replace(ent.text, ent.label_+'_')
    tokens = [d.text for d in doc]
#     [ent.merge(ent.root.tag_, ent.text, ent.label_) for ent in doc.ents]
#     return compact([d.ent_iob_+'-'+d.ent_type_ if d.ent_iob_ != 'O' else d.text for d in doc ]), tokens
#     return text, tokens
    temp = [d.ent_type_+'_' if d.ent_iob_ != 'O' else d.text for d in doc]
    return [x[0] for x in groupby(temp)], tokens

def get_split():
    rand = np.random.uniform()
    if rand > 0.95:
        split = 'test'
    #             test_num += 1
    elif rand > 0.91 and rand < 0.95:
        split = 'val'
    #         val_num += 1
    else:
        split = 'train'
    #         train_num += 1
    return split

if __name__ == '__main__':
    np.random.seed(42)
    nlp = spacy.load('en', disable=['parser', 'tagger'])
    print('Loading spacy modules.')
    news_data = []
    counter = 0
    test_num, val_num, train_num = 0, 0, 0

    print('Loading the json.')
    with open("../data/captioning_dataset.json", "rb") as f:
        captioning_dataset = json.load(f)

    for k, anns in tqdm.tqdm(captioning_dataset.items()):

        for ix, img in anns['images'].items():
            try:
                split = get_split()

                #         import ipdb; ipdb.set_trace()
                img = preprocess_sentence(img)
                template, full = NER(' '.join(img))
                if len(' '.join(template)) != 0:
                    news_data.append({'filename': k + '_' + ix + '.jpg', 'filepath': 'resized', 'cocoid': counter,
                                      'imgid': k + '_' + ix, 'sentences': [], 'sentences_full': [],
                                      #                               'sentences_article':[],
                                      'split': split})
                    news_data[counter]['sentences'].append(
                        {'imgid': counter, 'raw': ' '.join(template), 'tokens': template})
                    news_data[counter]['sentences_full'].append(
                        {'imgid': counter, 'raw': ' '.join(full), 'tokens': full})
                    counter += 1
            except:
                print img
    split_to_ix = {i:n['split'] for i, n in enumerate(news_data)}
    # train = [news_data[k] for k, v in split_to_ix.items() if v =='train']
    val = [news_data[k] for k, v in split_to_ix.items() if v =='val']
    test = [news_data[k] for k, v in split_to_ix.items() if v =='test']
    with open("../data/test.json", "wb") as f:
        json.dump(test, f)
    with open("../data/val.json", "wb") as f:
        json.dump(val, f)
    with open("../data/news_dataset.json", "wb") as f:
        json.dump(news_data, f)