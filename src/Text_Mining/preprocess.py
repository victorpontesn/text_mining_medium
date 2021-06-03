"""
    Module with specialized instructions for preprocess text data.
    By: Victor Pontes (victoraleff@gmail.com)
"""

import string
import re
from unidecode import unidecode
from .stopwords import stop_words
from bs4 import BeautifulSoup


class TextCleaner():

    def __init__(self):

        punctuation_chars = string.punctuation
        punctuation = dict(zip(punctuation_chars, [''] * len(punctuation_chars)))
        self.punctuation_dict = dict((re.escape(k), v) for k, v in punctuation.items()) 
        self.pattern_punctuation = re.compile("|".join(self.punctuation_dict.keys()))

        self.pattern_alpha = re.compile("\d+")

        stopwords = [' '+ word + ' ' for word in stop_words] 
        stopwords = dict(zip(stopwords, [' '] * len(stopwords)))
        self.stopwords_dict = dict((re.escape(k), v) for k, v in stopwords.items()) 
        self.pattern_stopwords = re.compile("|".join(self.stopwords_dict.keys()))

        html_remnants = ['&nbsp', '&amp','&quot', '&gt','&lt','&le','&ge',
                         '¿', '\r', '\n']
        html_ = dict(zip(html_remnants, [''] * len(html_remnants)))
        self.html_dict = dict((re.escape(k), v) for k, v in html_.items())
        self.pattern_html = re.compile("|".join(self.html_dict.keys())) 

        remnants_chars = ['--', "'", '''"''', '/']
        remnants = dict(zip(remnants_chars, [''] * len(remnants_chars)))
        self.remnants_dict = dict((re.escape(k), v) for k, v in remnants.items()) 
        self.pattern_remnants = re.compile("|".join(self.remnants_dict.keys()))

    def strip_punctuation(self, text: str)-> str:
        text = self.pattern_punctuation.sub(lambda m: self.punctuation_dict[re.escape(m.group(0))], text)
        return text

    @staticmethod
    def remove_accents(text: str)-> str:
        return unidecode(text)

    def strip_numbers(self, text: str)-> str:
        text = self.pattern_alpha.sub('', text)
        return  text
    
    def strip_remnants(self, text: str)-> str:
        text = self.pattern_remnants.sub('', text)
        return  text

    def strip_stopwords(self, text):
        text = ' ' + text + ' '
        text = self.pattern_stopwords.sub(lambda m: self.stopwords_dict[re.escape(m.group(0))], text)
        return text[1:-1]
        
    def cleaner_html(self, text: str)-> str:
        soup = BeautifulSoup(text, 'lxml')
        for s in soup(['script', 'style']):        
            s.decompose()

        text = ' '.join(soup.stripped_strings)
        text = self.pattern_html.sub(lambda m: self.html_dict[re.escape(m.group(0))], text)
        return text

    def transform(self, text, punctuation=True, accents=True, numbers=True, stopwords=True, html=True, lower=True):
        if html: text = self.cleaner_html(text)
        if stop_words: text = self.strip_stopwords(text)
        if punctuation: text = self.strip_punctuation(text)
        if accents: text = self.remove_accents(text)
        if numbers: text = self.strip_numbers(text)
        if lower: text = text.lower()
        text = self.strip_remnants(text)
        return text
