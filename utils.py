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


def tweet2tensor(tweet, vocab_dict, unk_token='__UNK__', verbose=False):
    '''
    Input: 
        tweet - A string containing a tweet
        vocab_dict - The words dictionary
        unk_token - The special string for unknown tokens
        verbose - Print info durign runtime
    Output:
        tensor - A python list withunique integer IDs
        representing the processed tweet
        
    '''

    words = process_tweet(tweet)

    tensor = []

    unk_ID = vocab_dict[unk_token]

    for word in words:
        word_ID = vocab_dict[word] if word in vocab_dict else unk_ID
        tensor.append(word_ID) 


    return tensor