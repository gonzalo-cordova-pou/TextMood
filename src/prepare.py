import utils as u
from trax.supervised import training
import json
import pandas as pd
import numpy as np


def preparation():
    # ================ #
    # DATA INGESTION #
    # ================ #

    # Load positive and negative tweets
    all_positive_tweets, all_negative_tweets = u.provisional_load_tweets()


    # ================ #
    # DATA PREPARATION #
    # ================ #

    # Split positive set into validation and training
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

    # Note that we build vocab using training data
    for tweet in train_x: 
        processed_tweet = u.process_tweet(tweet)
        for word in processed_tweet:
            if word not in Vocab: 
                Vocab[word] = len(Vocab)
    
    return train_pos, train_neg, val_pos, val_neg, train_x, val_x, train_y, val_y, Vocab



