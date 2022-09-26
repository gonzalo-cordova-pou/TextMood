import pandas as pd
from pathlib import Path
import numpy as np
import re
import json
import nltk
import string
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples 
import pandas as pd
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

data_path = dvc.api.get_url('training.1600000.processed.noemoticon.csv')

    # Reading the dataset with no columns titles and with latin encoding 
df = pd.read_csv(data_path, sep = ",", encoding='latin-1', header=None, error_bad_lines=False)

# As the data has no column titles, we will add our own
df.columns = ["label", "time", "date", "query", "username", "text"]

# Separating positive and negative rows
df_pos = df[df['label'] == 4]
df_neg = df[df['label'] == 0]
    
    # Only retaining 1/4th of our data from each output group
    # Feel free to alter the dividing factor depending on your workspace
    # 1/64 is a good place to start if you're unsure about your machine's power
df_pos = df_pos.iloc[:int(len(df_pos)/4)]
df_neg = df_neg.iloc[:int(len(df_neg)/4)]
print(len(df_pos), len(df_neg))

all_positive_tweets = df_pos.text.to_list()
all_negative_tweets = df_neg.text.to_list()

val_pos   = all_positive_tweets[40000:80000] # generating validation set for positive tweets
train_pos  = all_positive_tweets[:40000]# generating training set for positive tweets

# Split negative set into validation and training
val_neg   = all_negative_tweets[40000:80000] # generating validation set for negative tweets
train_neg  = all_negative_tweets[:40000] # generating training set for nagative tweets

# Delete all_positive_tweets and all_negative_tweets from memory
del all_positive_tweets
del all_negative_tweets

# Combine training data into one set
train_x = train_pos + train_neg 

# Combine validation data into one set
val_x  = val_pos + val_neg

# Set the labels for the training set (1 for positive, 0 for negative)
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

# Set the labels for the validation set (1 for positive, 0 for negative)
val_y  = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))


# Build the vocabulary

# Include special tokens 
# started with pad, end of line and unk tokens
Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 
stopwords_english = stopwords.words('english')

# Note that we build vocab using training data
for tweet in train_x: 
    processed_tweet = process_tweet(tweet)
    for word in processed_tweet:
        if word not in Vocab: 
            Vocab[word] = len(Vocab)

# Path of the output data folder
Path("data/processed").mkdir(exist_ok=True)
prepared_folder_path = Path("data/processed")

X_train_path = prepared_folder_path / "X_train.txt"
y_train_path = prepared_folder_path / "y_train.txt"
X_valid_path = prepared_folder_path / "X_valid.txt"
y_valid_path = prepared_folder_path / "y_valid.txt"
train_pos_path = prepared_folder_path / "train_pos.txt"
train_neg_path = prepared_folder_path / "train_neg.txt"
vocab_path = prepared_folder_path / "vocab.json"
val_pos_path = prepared_folder_path / "val_pos.txt"
val_neg_path = prepared_folder_path / "val_neg.txt"

with open(X_train_path, 'w', encoding = 'utf-8') as temp_file:
    for item in train_x:
        temp_file.write("%s\n" % item)

with open(y_train_path , 'w', encoding = 'utf-8') as temp_file:
    for item in train_y:
        temp_file.write("%s\n" % item)

with open(X_valid_path, 'w', encoding = 'utf-8') as temp_file:
    for item in val_x:
        temp_file.write("%s\n" % item)

with open(y_valid_path, 'w', encoding = 'utf-8') as temp_file:
    for item in val_y:
        temp_file.write("%s\n" % item)
        
with open(train_pos_path, 'w', encoding = 'utf-8') as temp_file:
    for item in train_pos:
        temp_file.write("%s\n" % item)

with open(train_neg_path, 'w', encoding = 'utf-8') as temp_file:
    for item in train_neg:
        temp_file.write("%s\n" % item)

with open(vocab_path, 'w', encoding = 'utf-8') as fp:
    json.dump(Vocab, fp)     

with open(val_pos_path, 'w', encoding = 'utf-8') as temp_file:
    for item in val_pos:
        temp_file.write("%s\n" % item)
        
with open(val_neg_path, 'w', encoding = 'utf-8') as temp_file:
    for item in val_neg:
        temp_file.write("%s\n" % item)