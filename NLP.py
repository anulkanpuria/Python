import os
import gdown
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re


nltk.download('punkt')


def download_assignment():
    folder_id = "1ltdsXAS_zaZ3hI-q9eze_QCzHciyYAJY"
    gdown.download_folder(id=folder_id, quiet=False, use_cookies=False)


def scrape_data():
    df = pd.read_excel('20211030 Test Assignment/Input.xlsx', header=0)
    url_list = df['URL']
    title = []
    body = []
    for link in url_list:
        source = requests.get(link)
        soup = BeautifulSoup(source.content, "html.parser")
        body0 = soup.findAll('div', class_='td-post-content tagdiv-type')
        body1 = soup.findAll('div', class_='td_block_wrap tdb_single_content tdi_130 td-pb-border-top '
                                           'td_block_template_1 td-post-content tagdiv-type')
        if soup.h3.text == 'Ooops... Error 404':
            title += [soup.h3.text]
            body += [soup.h3]
        elif len(body0) > 0:
            title += [soup.h1.text]
            body += [body0]
        else:
            title += [soup.h1.text]
            body += [body1]
    return title, body


def tokenize(title, content):
    tokens = []
    for title, content in zip(title, content):
        title = nltk.word_tokenize(title)
        token = [nltk.word_tokenize(cont.text) for cont in content][0]
        token = title + token
        token = [words for words in token if words.isalpha()]
        tokens += [token]
    return tokens


def count_sentences(title, content):
    count_sent = []
    for title, content in zip(title, content):
        sentences = [nltk.sent_tokenize(cont.text) for cont in content][0]
        sentence_in_title = nltk.sent_tokenize(title)
        sentences = sentence_in_title + sentences
        count_sent += [len(sentences)]
    return count_sent


def clean_stopwords(tokens):
    clean_tokens = []
    stopwords_list = []
    for file in os.listdir("20211030 Test Assignment/StopWords"):
        with open("20211030 Test Assignment/StopWords/" + file, 'r') as f:
            stopwords_list += f.read().split('\n')

    stopwords_list = [words.split('|')[0].lower().strip() for words in stopwords_list]

    for token in tokens:
        clean_token = [words.lower() for words in token if words.lower() not in stopwords_list]
        clean_tokens += [clean_token]

    return clean_tokens


def positive_words(cln_token):
    positive_word = []
    with open("20211030 Test Assignment/MasterDictionary/positive-words.txt", 'r') as f:
        positive_word += f.read().split('\n')

    count = []
    for article in cln_token:
        cnt = 0
        for words in article:
            if words in positive_word:
                cnt += 1
        count += [cnt]
    return count


def negative_words(cln_token):
    negative_word = []
    with open("20211030 Test Assignment/MasterDictionary/negative-words.txt", 'r') as f:
        negative_word += f.read().split('\n')

    count = []
    for article in cln_token:
        cnt = 0
        for words in article:
            if words in negative_word:
                cnt += 1
        count += [cnt]
    return count


def complex_words(cln_token):
    syllables = []
    for words in cln_token:
        syl = [nltk.SyllableTokenizer().tokenize(cont) for cont in words]
        syl = [words for words in syl if len(words) > 1]
        syllables += [len(syl)]
    return syllables


def syllable_count(cln_token):
    count = []
    syllables = []
    for words in cln_token:
        syl = [nltk.SyllableTokenizer().tokenize(cont) for cont in words]
        syl = [words for words in syl if len(words) > 1 and words[-1].endswith(('es', 'ed'))]
        syllables += [syl]

    for words in syllables:
        count += [len(words)]
    return count


def word_count():
    count = []
    clean_tokens = []
    titles, bodies = scrape_data()
    tokens = tokenize(titles, bodies)
    stop_words = list(stopwords.words('english'))
    for token in tokens:
        clean_token = [words.lower() for words in token if words.lower() not in stop_words]
        count += [len(clean_token)]
    return count


def personal_pronouns(tokens):
    count = []
    for articles in tokens:
        cnt = 0
        for words in articles:
            if re.search("^I$|^we$|^my$|^ours$|^us$", words):
                cnt += 1
        count += [cnt]
    return count


def char_sum(tokens):
    sum_list = []
    for articles in tokens:
        cnt = sum([len(words) for words in articles])
        sum_list += [cnt]
    return sum_list


def create_excel():
    titles, bodies = scrape_data()
    tokens = tokenize(titles, bodies)
    no_words = [len(words) for words in tokens]
    clean_words = clean_stopwords(tokens)
    pve_words = positive_words(clean_words)
    nve_words = negative_words(clean_words)
    tw_clean = [len(words) for words in clean_words]
    sent_no = count_sentences(titles, bodies)
    cx_word = complex_words(tokens)
    word_cnt = word_count()
    syl_count = syllable_count(tokens)
    pronouns = personal_pronouns(tokens)
    sum_char = char_sum(tokens)

    input = pd.read_excel('20211030 Test Assignment/Input.xlsx', header=0)
    columns = {
        'POSITIVE SCORE': pve_words,
        'NEGATIVE SCORE': nve_words,
        'WORD NOS': no_words,
        'TW CLEAN': tw_clean,
        'SENT NO': sent_no,
        'COMPLEX WORD COUNT': cx_word,
        'WORD COUNT': word_cnt,
        'SYLLABLE PER WORD': syl_count,
        'PERSONAL PRONOUNS': pronouns,
        'CHAR SUM': sum_char
    }

    df = pd.DataFrame(columns)
    df['POLARITY SCORE'] = (df['POSITIVE SCORE'] - df['NEGATIVE SCORE']) / (
                (df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) + 0.000001)
    df['SUBJECTIVITY SCORE'] = (df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) / (df['TW CLEAN'] + 0.000001)
    df['AVG SENTENCE LENGTH'] = df['WORD NOS'] / df['SENT NO']
    df['PERCENTAGE OF COMPLEX WORDS'] = df['COMPLEX WORD COUNT'] / df['WORD NOS']
    df['FOG INDEX'] = 0.4 * (df['AVG SENTENCE LENGTH'] + df['PERCENTAGE OF COMPLEX WORDS'])
    df['AVG NUMBER OF WORDS PER SENTENCE'] = df['WORD NOS'] / df['SENT NO']
    df['AVG WORD LENGTH'] = df['CHAR SUM'] / df['WORD NOS']
    df = input.join(df, how='inner')
    needed_cols = ['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
                   'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                   'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
    df = df[['URL_ID','URL']+needed_cols]
    df.loc[df['POSITIVE SCORE'] == 0, needed_cols] = 0
    df.to_excel('blackcoffer.xlsx', header=True, index=False)


if __name__ == "__main__":
    download_assignment()
    #create_excel()
