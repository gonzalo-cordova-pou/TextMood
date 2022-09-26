import utils as u
import random as rnd
import os
import numpy as np
import trax
from trax.supervised import training
import json
import mlflow
import pandas as pd
import our_model as cl
import prepare as pr
from mlflow import pyfunc
from trax import fastmath
# import trax.layers
from trax import layers as tl
train_pos, train_neg, val_pos, val_neg, train_x, val_x, train_y, val_y, Vocab = load_data()

def load_data():
    prepared_folder_path = Path("data/processed")
    X_train_path = prepared_folder_path / "X_train.txt"
    y_train_path = prepared_folder_path / "y_train.txt"
    X_valid_path = prepared_folder_path / "X_valid.txt"
    y_valid_path = prepared_folder_path / "y_valid.txt"
    train_pos_path = prepared_folder_path / "train_pos.txt"
    train_neg_path = prepared_folder_path / "train_neg.txt"
    val_pos_path = prepared_folder_path / "val_pos.txt"
    val_neg_path = prepared_folder_path / "val_neg.txt"
    vocab_path = prepared_folder_path / "vocab.json"

    train_x = open(X_train_path, encoding = 'utf-8').readlines()
    for i in range(len(train_x)):
        train_x[i] = train_x[i].replace('\n', '')

    val_x = open(X_valid_path, encoding = 'utf-8').readlines()
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
    Vocab = json.load(json_file)

    return train_pos, train_neg, val_pos, val_neg, train_x, val_x, train_y, val_y, Vocab 

print("Length train_pos: ", len(train_pos))
print("Length train_neg: ", len(train_neg))
print("Length val_pos: ", len(val_pos))
print("Length val_neg: ", len(val_neg))
print("Length train_x: ", len(train_x))
print("Length val_x: ", len(val_x))
print("Length train_y: ", len(train_y))
print("Length val_y: ", len(val_y))
print("Length Vocab: ", len(Vocab))

# ================ #
# TRAINING SETUP #
# ================ #

# Create the training data generator
def train_generator(batch_size, shuffle = False):
    return u.data_generator(train_pos, train_neg, batch_size, True, Vocab, shuffle)

# Create the validation data generator
def val_generator(batch_size, shuffle = False):
    return u.data_generator(val_pos, val_neg, batch_size, True, Vocab, shuffle)

# Set the random number generator for the shuffle procedure
rnd.seed(30) 

print("####### CHECKPOINT 1 ########")
# Get a batch from the train_generator and inspect.
inputs, targets, example_weights = next(train_generator(4, shuffle=True))
print("####### CHECKPOINT 2 ########")

np = fastmath.numpy

# use the fastmath.random module from trax
random = fastmath.random

tmp_key = random.get_prng(seed=1)
tmp_shape=(2,3)

# Generate a weight matrix
# Note that you'll get an error if you try to set dtype to tf.float32, where tf is tensorflow
# Just avoid setting the dtype and allow it to use the default data type
tmp_weight = trax.fastmath.random.normal(key=tmp_key, shape=tmp_shape)

print("Weight matrix generated with a normal distribution with mean 0 and stdev of 1")
#display(tmp_weight)

tmp_embed = tl.Embedding(vocab_size=3, d_feature=2)
#display(tmp_embed)

# ================ #
# MODEL TRAINING #
# ================ # 

tmp_model = cl.classifier(len(Vocab))

batch_size = 2
rnd.seed(271)

print("####### CHECKPOINT 3 ########")

mlflow.tensorflow.autolog()

with mlflow.start_run():

    train_task = training.TrainTask(
        labeled_data=train_generator(batch_size=batch_size, shuffle=True),
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=10,
    )

    eval_task = training.EvalTask(
        labeled_data=val_generator(batch_size=batch_size, shuffle=True),
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )

    model = cl.classifier(len(Vocab))


    output_dir = '../models/'

    #mlflow.log_artifacts("./model")

    print("####### CHECKPOINT 4 ########")

    training_loop = u.train_model(model, train_task, eval_task, 100, output_dir)

    print("####### CHECKPOINT 5 ########")

    #pyfunc_model = pyfunc.load_model(mlflow.get_artifact_uri("model"))




# Create the validation data generator
def val_generator(batch_size, shuffle = False):
    return u.data_generator(val_pos, val_neg, batch_size, True, Vocab, shuffle)

# Create the validation data generator
def test_generator(batch_size, shuffle = False):
    return u.data_generator(val_pos, val_neg, batch_size, False, Vocab, shuffle)


# ================ #
# MODEL EVALUATION #
# ================ # 

# test your function
tmp_val_generator = val_generator(64)


# get one batch
tmp_batch = next(tmp_val_generator)

# Position 0 has the model inputs (tweets as tensors)
# position 1 has the targets (the actual labels)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch

# feed the tweet tensors into the model to get a prediction
tmp_pred = training_loop.eval_model(tmp_inputs)

tmp_acc, tmp_num_correct, tmp_num_predictions = u.compute_accuracy(preds=tmp_pred, y=tmp_targets, y_weights=tmp_example_weights)

print(f"Model's prediction accuracy on a single training batch is: {100 * tmp_acc}%")
print(f"Weighted number of correct predictions {tmp_num_correct}; weighted number of total observations predicted {tmp_num_predictions}")

# ================ #
# MODEL EVALUATION IN TEST DATA#
# ================ # 

# testing the accuracy of your model: this takes around 20 seconds
model = training_loop.eval_model
accuracy = u.test_model(test_generator(16), model)

print(f'The accuracy of your model on the validation set is {accuracy:.4f}', )


# ================ #
# MODEL EVALUATION WITH NEW DATA #
# ================ # 

# try a positive sentence
sentence = "It's such a nice day, think i'll be taking Sid to Ramsgate fish and chips for lunch at Peter's fish factory and then the beach maybe"
tmp_pred, tmp_sentiment = u.predict(sentence, Vocab, model)
print(f"The sentiment of the sentence \n***\n\"{sentence}\"\n***\nis {tmp_sentiment}.")

print()
# try a negative sentence
sentence = "I hated my day, it was the worst, I'm so sad."
tmp_pred, tmp_sentiment = u.predict(sentence, Vocab, model)
print(f"The sentiment of the sentence \n***\n\"{sentence}\"\n***\nis {tmp_sentiment}.")

'''
# ================ #
# MODEL PREDICTION #
# ================ # 

# Create a generator object
tmp_train_generator = train_generator(16)

# get one batch
tmp_batch = next(tmp_train_generator)

# Position 0 has the model inputs (tweets as tensors)
# position 1 has the targets (the actual labels)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch

# feed the tweet tensors into the model to get a prediction
tmp_pred = training_loop.eval_model(tmp_inputs)

# turn probabilites into category predictions
tmp_is_positive = tmp_pred[:,1] > tmp_pred[:,0]
for i, p in enumerate(tmp_is_positive):
    print(f"Neg log prob {tmp_pred[i,0]:.4f}\tPos log prob {tmp_pred[i,1]:.4f}\t is positive? {p}\t actual {tmp_targets[i]}")
'''
