import string
import re
import os
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples
import pandas as pd
import numpy as np
import random as rnd
from trax.supervised import training
from pathlib import Path

print("Loading TweetTokenizer...")
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
print("Finsihed loading TweetTokenizer")

def load_data():
    '''Read and load data'''
    prepared_folder_path = Path("data/processed")
    x_train_path = prepared_folder_path / "X_train.txt"
    y_train_path = prepared_folder_path / "y_train.txt"
    x_valid_path = prepared_folder_path / "X_valid.txt"
    y_valid_path = prepared_folder_path / "y_valid.txt"
    train_pos_path = prepared_folder_path / "train_pos.txt"
    train_neg_path = prepared_folder_path / "train_neg.txt"
    val_pos_path = prepared_folder_path / "val_pos.txt"
    val_neg_path = prepared_folder_path / "val_neg.txt"
    vocab_path = prepared_folder_path / "vocab.json"

    train_x = open(x_train_path, encoding = 'utf-8').readlines()
    for i in range(len(train_x)):
        train_x[i] = train_x[i].replace('\n', '')

    val_x = open(x_valid_path, encoding = 'utf-8').readlines()
    for i in range(len(val_x)):
        val_x[i] = val_x[i].replace('\n', '')

    train_pos = open(train_pos_path, encoding = 'utf-8').readlines()
    for i in range(len(train_pos)):
        train_pos[i] = train_pos[i].replace('\n', '')

    train_neg = open(train_neg_path, encoding = 'utf-8').readlines()
    for i in range(len(train_neg)):
        train_neg[i] = train_neg[i].replace('\n', '')

    val_neg = open(val_neg_path, encoding = 'utf-8').readlines()
    for i in range(len(val_neg)):
        val_neg[i] = val_neg[i].replace('\n', '')

    val_pos = open(val_pos_path, encoding = 'utf-8').readlines()
    for i in range(len(val_pos)):
        val_pos[i] = val_pos[i].replace('\n', '')

    val_y = open(y_valid_path, encoding = 'utf-8').readlines()
    for i in range(len(val_y)):
        val_y[i] = float(List[i])
    val_y = np.array(val_y)

    train_y = open(y_train_path, encoding = 'utf-8').readlines()
    for i in range(len(train_y)):
        val_y[i] = float(List[i])
    train_y = np.array(train_y)

    json_file = open(vocab_path, 'r', encoding = 'utf-8')
    vocab = json.load(json_file)

    return train_pos, train_neg, val_pos, val_neg, train_x, val_x, train_y, val_y, vocab 


print("Loading stopwords and stemmer...")
# I'm not sure we want to get into stop words
stopwords_english = stopwords.words('english')
# Also have my doubts about stemming...
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print("Finished loading stopwords and stemmer")

def provisional_load_tweets(size=10000):
    '''Provisional data'''
    df_raw = pd.read_csv('./data/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header=None)
    # As the data has no column titles, we will add our own
    df_raw.columns = ["label", "time", "date", "query", "username", "text"]

    df = df_raw[['label', 'text']]

    # Separating positive and negative rows
    df_pos = df[df['label'] == 4]
    df_neg = df[df['label'] == 0]
    
    # Only retaining 1/4th of our data from each output group
    # Feel free to alter the dividing factor depending on your workspace
    # 1/64 is a good place to start if you're unsure about your machine's power
    df_pos = df_pos.iloc[:int(size/2)]
    df_neg = df_neg.iloc[:int(size/2)]
    print(len(df_pos), len(df_neg))

    all_positive_tweets = df_pos.text.to_list()
    all_negative_tweets = df_neg.text.to_list()

    return all_positive_tweets, all_negative_tweets

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

    unk_id = vocab_dict[unk_token] # unknown token id

    for word in words:

        # Get the unique integer id of each word

        word_id = vocab_dict[word] if word in vocab_dict else unk_id
        tensor.append(word_id)


    return tensor

def data_generator(data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False):
    '''
    Input: 
        data_pos - Set of posstive examples
        data_neg - Set of negative examples
        batch_size - number of samples per batch. Must be even
        loop - True or False
        vocab_dict - The words dictionary
        shuffle - Shuffle the data order
    Yield:
        inputs - Subset of positive and negative examples
        targets - The corresponding labels for the subset
        example_weights - An array specifying the importance of each example    
    '''     
    
    # make sure the batch size is an even number
    # to allow an equal number of positive and negative samples
    assert batch_size % 2 == 0
    
    # Number of positive examples in each batch is half of the batch size
    # same with number of negative examples in each batch
    n_to_take = batch_size // 2
    
    # Use pos_index to walk through the data_pos array
    # same with neg_index and data_neg
    pos_index = 0
    neg_index = 0
    
    len_data_pos = len(data_pos)
    len_data_neg = len(data_neg)
    
    # Get and array with the data indexes
    pos_index_lines = list(range(len_data_pos))
    neg_index_lines = list(range(len_data_neg))
    
    # shuffle lines if shuffle is set to True
    if shuffle:
        rnd.shuffle(pos_index_lines)
        rnd.shuffle(neg_index_lines)
        
    stop = False
    
    # Loop indefinitely
    while not stop: 
        
        # create a batch with positive and negative examples
        batch = []
        
        # First part: Pack n_to_take positive examples
        # Start from pos_index and increment i up to n_to_take
        for i in range(n_to_take):
                    
            # If the positive index goes past the positive dataset lenght,
            if pos_index >= len_data_pos: 
                
                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True;
                    break;
                
                # If user wants to keep re-using the data, reset the index
                pos_index = 0
                
                if shuffle:
                    # Shuffle the index of the positive sample
                    rnd.shuffle(pos_index_lines)
                    
            # get the tweet as pos_index
            tweet = data_pos[pos_index_lines[pos_index]]
            
            # convert the tweet into tensors of integers representing the processed words
            tensor = tweet2tensor(tweet, vocab_dict)
            
            # append the tensor to the batch list
            batch.append(tensor)
            
            # Increment pos_index by one
            pos_index = pos_index + 1

        # Second part: Pack n_to_take negative examples

        # Using the same batch list, start from neg_index and increment i up to n_to_take
        for i in range(n_to_take):
            
            # If the negative index goes past the negative dataset length,
            if neg_index >= len_data_neg:
                
                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True;
                    break;
                    
                # If user wants to keep re-using the data, reset the index
                neg_index = 0
                
                if shuffle:
                    # Shuffle the index of the negative sample
                    rnd.shuffle(neg_index_lines)
            # get the tweet as pos_index
            tweet = data_neg[neg_index_lines[neg_index]]
            
            # convert the tweet into tensors of integers representing the processed words
            tensor = tweet2tensor(tweet, vocab_dict)
            
            # append the tensor to the batch list
            batch.append(tensor)
            
            # Increment neg_index by one
            neg_index += 1

        if stop:
            break;

        # Update the start index for positive data 
        # so that it's n_to_take positions after the current pos_index
        pos_index += n_to_take
        
        # Update the start index for negative data 
        # so that it's n_to_take positions after the current neg_index
        neg_index += n_to_take
        
        # Get the max tweet length (the length of the longest tweet) 
        # (you will pad all shorter tweets to have this length)
        max_len = max([len(t) for t in batch]) 
        
        
        # Initialize the input_l, which will 
        # store the padded versions of the tensors
        tensor_pad_l = []
        # Pad shorter tweets with zeros
        for tensor in batch:

            # Get the number of positions to pad for this tensor so that it will be max_len long
            n_pad = max_len - len(tensor)
            
            # Generate a list of zeros, with length n_pad
            pad_l = [0]*n_pad
            
            # concatenate the tensor and the list of padded zeros
            tensor_pad = tensor + pad_l
            
            # append the padded tensor to the list of padded tensors
            tensor_pad_l.append(tensor_pad)

        # convert the list of padded tensors to a numpy array
        # and store this as the model inputs
        inputs = np.array(tensor_pad_l)
  
        # Generate the list of targets for the positive examples (a list of ones)
        # The length is the number of positive examples in the batch
        target_pos = [1]*n_to_take
        
        # Generate the list of targets for the negative examples (a list of ones)
        # The length is the number of negative examples in the batch
        target_neg = [0]*n_to_take
        
        # Concatenate the positve and negative targets
        target_l = target_pos + target_neg
        
        # Convert the target list into a numpy array
        targets = np.array(target_l)

        # Example weights: Treat all examples equally importantly.
        example_weights = np.ones_like(targets)
        
        # note we use yield and not return
        yield inputs, targets, example_weights

def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    '''
    Input:
        classifier - the model you are building
        train_task - Training task
        eval_task - Evaluation task
        n_steps - the evaluation steps
        output_dir - folder to save your files
    Output:
        trainer -  trax trainer
    '''

    training_loop = training.Loop(
                                classifier, # The learning model
                                train_task, # The training task
                                eval_tasks = [eval_task], # The evaluation task
                                output_dir = output_dir # The output directory
    )

    training_loop.run(n_steps = n_steps)

    # Return the training_loop, since it has the model.
    return training_loop

def compute_accuracy(preds, y, y_weights):
    """
    Input: 
        preds: a tensor of shape (dim_batch, output_dim)
        y: a tensor of shape (dim_batch, output_dim) with the true labels
        y_weights: a n.ndarray with the a weight for each example
    Output: 
        accuracy: a float between 0-1
        weighted_num_correct (np.float32): Sum of the weighted correct predictions
        sum_weights (np.float32): Sum of the weights
    """
    
    # Create an array of booleans,
    # True if the probability of positive sentiment is greater than
    # the probability of negative sentiment
    # else False
    is_pos =  preds[:, 1] > preds[:, 0]

    # convert the array of booleans into an array of np.int32
    is_pos_int = is_pos.astype(np.int32)
    
    # compare the array of predictions (as int32) with the target (labels) of type int32
    correct = is_pos_int == y

    # Count the sum of the weights.
    sum_weights = np.sum(y_weights)
    
    # convert the array of correct predictions (boolean) into an arrayof np.float32
    correct_float = correct.astype(np.float32)
    
    # Multiply each prediction with its corresponding weight.
    weighted_correct_float = correct_float * y_weights

    # Sum up the weighted correct predictions (of type np.float32), to go in the
    # denominator.
    weighted_num_correct = np.sum(weighted_correct_float)
 
    # Divide the number of weighted correct predictions by the sum of the
    # weights.
    accuracy = weighted_num_correct / sum_weights

    return accuracy, weighted_num_correct, sum_weights

def test_model(generator, model):
    '''
    Input: 
        generator: an iterator instance that provides batches of inputs and targets
        model: a model instance
    Output: 
        accuracy: float corresponding to the accuracy
    '''
    
    accuracy = 0.
    total_num_correct = 0
    total_num_pred = 0
    
    
    for batch in generator: 
        
        # Retrieve the inputs from the batch
        inputs = batch[0]
        
        # Retrieve the targets (actual labels) from the batch
        targets = batch[1]
        
        # Retrieve the example weight.
        example_weight = batch[2]

        # Make predictions using the inputs
        pred = model(inputs)
        
        # Calculate accuracy for the batch by comparing its predictions and targets
        batch_accuracy, batch_num_correct, batch_num_pred = compute_accuracy(pred, targets, example_weight)
        
        # Update the total number of correct predictions
        # by adding the number of correct predictions from this batch
        total_num_correct += batch_num_correct
        
        # Update the total number of predictions
        # by adding the number of predictions made for the batch
        total_num_pred += batch_num_pred

    # Calculate accuracy over all examples
    accuracy = total_num_correct / total_num_pred
    
    return accuracy

def predict(sentence, vocab, model):
    '''Predict on your own sentnece'''
    inputs = np.array(tweet2tensor(sentence, vocab_dict=vocab))
    
    # Batch size 1, add dimension for batch, to work with the model
    inputs = inputs[None, :]  
    
    # predict with the model
    preds_probs = model(inputs)
    
    # Turn probabilities into categories
    preds = int(preds_probs[0, 1] > preds_probs[0, 0])
    
    sentiment = "negative"
    if preds == 1:
        sentiment = 'positive'

    return preds, sentiment