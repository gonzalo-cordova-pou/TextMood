import string
import re
import os
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples 

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Stop words are messy and not that compelling; 
# "very" and "not" are considered stop words, but they are obviously expressing sentiment

# The porter stemmer lemmatizes "was" to "wa".  Seriously???

# I'm not sure we want to get into stop words
stopwords_english = stopwords.words('english')

# Also have my doubts about stemming...
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

