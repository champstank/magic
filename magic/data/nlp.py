"""This is for all NLP natural language processing needs"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('wordnet')  #<< make sure this is downloaded
nltk.download('stopwords')

def word_pipeline(word, lang = "english"):
    """This function preprocesses the text to get ready for count vectorizer
    Params:
        word string word to be processed
    Returns:
        word string processed word
    """
    stemmer = SnowballStemmer(lang)
    lemmatizer = WordNetLemmatizer()

    #word = BeautifulSoup(word,"html5lib").get_text()
    word = re.sub("[^a-zA-Z]", " ", word)
    word = word.lower()
    word = stemmer.stem(word)
    word = lemmatizer.lemmatize(word)
    if word in stopwords.words(lang):
        word = None
    return word

def process_words(doc):
    """This pipeline calls each word for count vectorizer"""
    analyzer = CountVectorizer().build_analyzer()
    return (word_pipeline(w) for w in analyzer(doc))


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
