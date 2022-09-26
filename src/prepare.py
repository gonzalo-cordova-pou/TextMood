
import pandas as pd
from pathlib import Path
import numpy as np

input_folder_path = Path("data/raw")
train_path = input_folder_path / "training.1600000.processed.noemoticon.csv"

    # Reading the dataset with no columns titles and with latin encoding 
df = pd.read_csv(train_path, sep = ",", encoding='latin-1', header=None, error_bad_lines=False)

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

val_pos   = all_positive_tweets[40000:] # generating validation set for positive tweets
train_pos  = all_positive_tweets[:40000]# generating training set for positive tweets

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

# Path of the output data folder
Path("data/processed").mkdir(exist_ok=True)
prepared_folder_path = Path("data/processed")

X_train_path = prepared_folder_path / "X_train.csv"
y_train_path = prepared_folder_path / "y_train.csv"
X_valid_path = prepared_folder_path / "X_valid.csv"
y_valid_path = prepared_folder_path / "y_valid.csv"

with open(X_train_path, 'w') as temp_file:
    for item in train_x:
        temp_file.write("%s\n" % item)

with open(y_train_path , 'w') as temp_file:
    for item in train_y:
        temp_file.write("%s\n" % item)

with open(X_valid_path, 'w') as temp_file:
    for item in val_x:
        temp_file.write("%s\n" % item)

with open(y_valid_path, 'w') as temp_file:
    for item in val_y:
        temp_file.write("%s\n" % item)

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
