from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


class Vectorizers(): 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs)
        
    def count_vectorizer(self,documents,n_gram_range,max_df,min_df,tokenizer,lowercase,input, max_features ):
        assert len(documents)>1
        self.vectorizer = CountVectorizer(ngram_range = n_gram_range, max_df = max_df, min_df = min_df, tokenizer = tokenizer, 
                                          lowercase = lowercase, input = input, max_features = max_features)
        vectorized = self.vectorizer.fit_transform(documents)
        vectorized = pd.DataFrame(vectorized.toarray(),columns = self.vectorizer.get_feature_names())
        return vectorized,self.vectorizer.vocabulary_,self.vectorizer.stop_words_
    
    def tfidf_vectorizer(self,documents,n_gram_range,max_df, min_df,use_idf,smooth_idf):
        assert len(documents)>1
        self.vectorizer = TfidfVectorizer(ngram_range = n_gram_range, max_df = max_df, min_df = min_df,use_idf = use_idf, smooth_idf = smooth_idf)
        vectorized = self.vectorizer.fit_transform(documents)
        vectorized = pd.DataFrame(vectorized.toarray(),columns = self.vectorizer.get_feature_names())
        
        if use_idf: 
            return vectorized, self.vectorizer.idf_,self.vectorizer.vocabulary_,self.vectorizer.stop_words_
        else: 
            return vectorized, self.vectorizer.vocabulary_,self.vectorizer.stop_words_