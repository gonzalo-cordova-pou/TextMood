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


def process_tweet(tweet):
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        clean_tweet: a list of words containing the processed tweet
    
    '''

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    clean_tweet = []
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
            word not in string.punctuation): # remove punctuation
            #clean_tweet.append(word)
            stem_word = stemmer.stem(word) # stemming word
            clean_tweet.append(stem_word)
    
    return clean_tweet


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

    # Process the tweet into a clean list of words
    words = process_tweet(tweet)

    tensor = []

    unk_ID = vocab_dict[unk_token] # unknown token id

    for word in words:

        # Get the unique integer id of each word

        word_ID = vocab_dict[word] if word in vocab_dict else unk_ID
        tensor.append(word_ID) 


    return tensor