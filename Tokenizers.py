import nltk 
from tqdm import tqdm_notebook
import inflect
import re 
import pandas as pd 

class Tokenizers(): 
    def __init__(self,**kwargs): 
        self.verbose = True
        self.__dict__.update(kwargs)
        
    def word_tokenizer(self,document): 
        return nltk.tokenize.word_tokenize(document)
        
    def punc_tokenizer(self,document):
        return nltk.tokenize.wordpunct_tokenize(document)
        
    def sentence_tokenizer(self,document): 
        return nltk.tokenize.sent_tokenize(document)
    
    def tokenize_documents(self,documents,tokenizer = "punc"):
        tokenizer = getattr(self,f"{tokenizer}_tokenizer")
        if self.verbose:
            return [[word for word in tokenizer(document) if word.isalnum()] for document in tqdm_notebook(documents)]
        else: 
            return [tokenizer(document) for document in documents]
        
    def numbers_to_word(self,documents): 
        p = inflect.engine()
        return [[p.number_to_words(word) if word.isnumeric() else word for word in document]for document in documents]
        
    def replace_words_regexp(self,documents,pattern,replace): 
        return [[re.sub(pattern,replace,word) for word in document] for document in tqdm_notebook(documents)]
    
    def word_to_string(self,documents): 
        return [" ".join(document) for document in documents]
    