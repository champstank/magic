# imports 
import re
import nltk
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import xgboost
from scipy.stats import pearsonr
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, precision_score, recall_score


#====================================  SUPPORTING stateless functions  ==============================================#
def is_regression(y):
    """
    This function check to see if target is a regression
    Params:
        y np.array label array
    Returns:
        True
    """
    try:
        y_float = np.array(np.array(y,dtype=int),dtype=float)
        return np.sum(np.abs(y_float - y))>0    # Is there an error with int conversion?
    except:
        return False                            # NOT regression, you through exception i.e. ['cat','dog']
    
def hashfile(filename):
    """
    This function will create a hash for a file based on file content
    Params:
        filename string filename to be hashed
    Returns:
        string hashed file information
    """
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

def export_model(model,results,filename):
    """
    This function will export model and results dictionary to disk for later use
    Params:
        model class model to be used
        basename string model name
        filename str filename being input
    Returns:
        True
    """
    basename = filename.split('.')[0]
    basename = basename.split('/')[-1]  # ignore folder names
    # calculate the hash on filename
    file_hash = hashfile(filename)
    new_filename = basename + '_' + file_hash + '.p'
    pickle.dump([model,results],open(new_filename,'wb'))
    return
      
#==============================================  CLASS  ===========================================================#
class dereksdocker():
    """This class will automate supervised regression & classification workflows"""
    
    def __init__(self,speed='fast',filename='None'):
        nltk.download('wordnet')  #<< make sure this is downloaded
        nltk.download('stopwords')
        
        self.filename = filename
        self.df = None
        self.speed = speed
        self.data_type = None
        self.analyzer = CountVectorizer().build_analyzer()
        self.lang = "english"
        self.stops = set(stopwords.words(self.lang))
        self.stemmer = SnowballStemmer(self.lang)
        self.lemmatizer = WordNetLemmatizer()
        self.X = None
        self.y = None
        self.macro_features = None
        self.micro_features = None
        self.results = {}
        self.model = None
        # run these below to make sure class is ready



    def file_2_df(self):
        """
        This function loads any file into a pandass dataframe. 
        Params:
            filename string i.e. file.csv, file.xls
        Returns:
            df pandas.dataframe 
        """
        if self.filename.split('.')[-1].lower()=='csv':
            print "csv detected"
            self.df = pd.read_csv(self.filename)
            self.data_type='csv'
        elif self.filename.split('.')[-1].lower()=='tsv':
            print 'text detected'
            df = pd.read_csv(self.filename, sep='\t')
            self.df = df[df.columns[::-1]]
            self.data_type='text'
            print self.df[:2]   #data frame preview
        elif self.filename.split('.')[-1].lower()=='xls':
            print "excel detected"
            self.df = pd.read_excel(self.filename)
            self.data_type='excel'
        else:
            print filename+": is not supported yet!"
        return self.df
    
    def word_pipeline(self,word):
        """This function preprocesses the text to get ready for count vectorizer
        Params:
            word string word to be processed
        Returns:
            word string processed word
        """
        word = BeautifulSoup(word,"html5lib").get_text()       
        word = re.sub("[^a-zA-Z]", " ", word) 
        word = word.lower() 
        word = self.stemmer.stem(word)
        word = self.lemmatizer.lemmatize(word)
        if word in self.stops:
            word = None
        return word

    def process_words(self,doc):
        """This pipeline calls each word for count vectorizer"""
        return (self.word_pipeline(w) for w in self.analyzer(doc))  
    
    def df_classifier_input(self):
        """
        This function will take a dataframe and split for binary
        Params:
            filename string i.e. file.csv, file.xls
        Returns:
            df pandas.dataframe 
        """
        df = self.df
        macro_features = df.columns[0:-1]     # grab column names
        target_label = df.columns[-1]         # grab target column name

        # Process input features
        if self.data_type=='text':  
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
            from scipy.sparse import hstack
            from scipy.sparse import csr_matrix

            print'This is a text, bag of words model'

            def text_processor(df,analyzer='word'):
                workflow = Pipeline([('vect', CountVectorizer(analyzer=analyzer)), ('tfidf',TfidfTransformer())])
                text_list = list(df[df.columns[0]].values)
                X = workflow.fit_transform(text_list)
                if analyzer is 'word':
                    macro_features = ['text']
                else:
                    macro_features = ['text_SL']   #SL = stemming & lemming
                #/////////////////////////////////
                cv = workflow.steps[0][1]
                vocab_dict = cv.vocabulary_
                id_vocab_dict={}
                for key in vocab_dict:
                    id_vocab_dict[str(vocab_dict[key])] = key 
                micro_features =[]
                for word in range(0,len(id_vocab_dict)):
                    micro_features.append(id_vocab_dict[str(word)])
                return X,micro_features,macro_features

            def process_sentiment(df):
                analyzer = SentimentIntensityAnalyzer()
                text_list = list(df[df.columns[0]].values)
                data=[]
                for msg in text_list:
                    vs = analyzer.polarity_scores(msg)
                    data.append([vs['neg'],vs['neu'],vs['pos'],vs['compound']])
                X = csr_matrix(np.array(data))
                micro_features = ['neg_sentiment','neu_sentiment','pos_sentiment','combined_sentiment']
                macro_features = ['text_sentiment']
                return X,micro_features,macro_features


            X,micro_features,macro_features = text_processor(df)
            
            if self.speed=='slow':
                X_2,micro_features_2,macro_features_2 = text_processor(df,analyzer=self.process_words)
                X_3,micro_features_3,macro_features_3 = process_sentiment(df)   #calculate sentiment
                #Combine all features
                micro_features.append(micro_features_2)
                micro_features.append(micro_features_3)
                macro_features.append(macro_features_2)
                macro_features.append(macro_features_3)
                X = csr_matrix(hstack((X,X_2,X_3))) # hstacking sparse matrices changes type to COO, changing to CSR

            #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        else:
            print 'NOT text'
            X_df_dummy = pd.get_dummies(df[macro_features])     
            micro_features = X_df_dummy.columns         # new columns names from tokenizer
            X = X_df_dummy.as_matrix()                        # Numpy array, from dataframe

        # Process Y
        y_df = df[target_label]
        y_raw = y_df.as_matrix()

        if is_regression(y_raw):
            print 'regression detected'
            y = y_raw
        else:
            print 'classification detected'
            y_df_dummy = pd.get_dummies(y_df)
            y = np.argmax(y_df_dummy.as_matrix(),axis=1)    

        self.X = X
        self.y = y
        self.macro_features = macro_features
        self.micro_features = micro_features
        return 

    def scoring(self,y,ypred_class,ypred_prob):
        """
        This function will create a scored dictionary of your predictions
        Params:
            y np.array actual value
            ypred_class np.array predicted values
            ypred_prob np.array predicted probabilities of values
        Returns:
            results dict dictionary of all results 
        """

        results={}
        if len(np.unique(y))>2:
            print('Multiclass detected!')
            results['accuracy'] = accuracy_score(y,ypred_class)
            results['confusion_matrix'] = confusion_matrix(y,ypred_class)
            results['observation_count'] = len(y)
        else:   # must be binary
            print('Binary detected!')
            results['auc_score'] = roc_auc_score(y,ypred_prob)
            results['accuracy'] = accuracy_score(y,ypred_class)
            results['confusion_matrix'] = confusion_matrix(y,ypred_class)
            results['pearson-r'] = pearsonr(y,ypred_prob)[0]
            results['pearson-r-pval'] = pearsonr(y,ypred_prob)[1]
            results['average_precision_score'] = average_precision_score(y,ypred_class)
            results['f1_score'] = f1_score(y,ypred_class)
            results['precision_score'] = precision_score(y,ypred_class)
            results['recall_score'] = recall_score(y,ypred_class)
            results['observation_count'] = len(y)
            results['label_balance'] = np.mean(y)
        self.results = results
        return self.results
    
    def train_and_validate(self):    
        """
        This function will train and cross validate your model
        Params:
            X np.array(matrix) input features
            y np.array(vector) target 
            model class this is the model that will be trained and tested
        Returns:
            model class trained model object
            scoring() function returns scored dictionary
        """

        X = self.X
        y = self.y
        model = self.model

        ypred_class = np.zeros_like(y,dtype=float)                     # initialize holder array, make sure it is float 
        ypred_prob = np.zeros_like(y,dtype=float)             
        if is_regression(y):
            from sklearn.model_selection import KFold
            skf = KFold(n_splits=10)
            for train_index, test_index in skf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train,y_train)
                ypred_class[test_index] = model.predict(X_test)
        else: # must be classification
            skf = StratifiedKFold(n_splits=10)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train,y_train)

                ypred_class[test_index] = model.predict(X_test)
                ypred_prob[test_index] = model.predict_proba(X_test)[:,1]  
        self.model = model
        return self.model, self.scoring(y,ypred_class,ypred_prob)
    
    def model_search(self):
        """
        This function will decide what type of model to use
        Params:
            model class trained model object
            results dict trained model scoring information
            filename str filename being input
        Returns:
            True
        """
        X = self.X
        y = self.y
        
        if self.speed=='fast':
            self.model = LogisticRegression()
        else:   # must be fast
            if self.data_type=='text':   # sparse matrix 
                self.model = xgboost.XGBClassifier()
            else:
                self.model = GradientBoostingClassifier()                        # define a model

            # HYPER PARAM TUNING
            # model = gridsearch()...asdf.asf.da
            print 'tuning parameters'
        #---------------------------------------------------------------------------------------------------#
            #param_grid = {
            #    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
            #}

            #model = GridSearchCV(model, param_grid=param_grid)
        #---------------------------------------------------------------------------------------------------#
            param_dist = {
                'learning_rate' : [0.0001, 0.001, 0.0015, 0.01, 0.1, 0.15, 0.02, 0.2, 0.03, 0.3]
            }
            self.model = RandomizedSearchCV(self.model, param_distributions=param_dist)

        return self.model
    
    def run(self):
        """
        This function will run entire workflow for prediction from file to model + results
        Params:
            filename str filename to be run
        Returns:
            True
        """

        start_time = time.time()
        self.file_2_df()                                    # load file into pandas dataframe
        self.df_classifier_input()  # prepare data for ML
        self.model_search()                         # model search data 

        train_model, results = self.train_and_validate()        # train model, cross validate and score
        for key in results:
            print "     ",key,":",results[key]

        export_model(train_model,results, self.filename)                 # save model to disk
        run_time = time.time() - start_time
        print "Ran in %.3f seconds" % run_time
        return True
