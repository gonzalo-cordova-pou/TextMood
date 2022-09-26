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

# Split negative set into validation and training
val_neg   = all_negative_tweets[40000:] # generating validation set for negative tweets
train_neg  = all_negative_tweets[:40000] # generating training set for nagative tweets

# Combine training data into one set
train_x = train_pos + train_neg 

# Combine validation data into one set
val_x  = val_pos + val_neg

# Set the labels for the training set (1 for positive, 0 for negative)
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

# Set the labels for the validation set (1 for positive, 0 for negative)
val_y  = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))

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