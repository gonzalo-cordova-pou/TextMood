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
import tensorflow as tf
# import trax.layers
from trax import layers as tl
train_pos, train_neg, val_pos, val_neg, train_x, val_x, train_y, val_y, Vocab = pr.preparation()

# Save Vocab to file
with open('Vocab.json', 'w') as fp:
    json.dump(Vocab, fp)

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

# Create the validation data generator
def test_generator(batch_size, shuffle = False):
    return u.data_generator(val_pos, val_neg, batch_size, False, Vocab, shuffle)

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

batch_size = 32
rnd.seed(271)

print("####### CHECKPOINT 3 ########")

mlflow.tensorflow.autolog()

with mlflow.start_run() as run:

    output_dir = '../models/{}/'.format(run.info.run_id)


    # Choose an optimizer and log it to mlflow
    lr = 0.01
    optimizer_name = "Adam" # choices are "Adam", "SGD"    
    if optimizer_name == "SGD":
        optimizer = trax.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = trax.optimizers.Adam(learning_rate=lr)
    mlflow.log_param("optimizer", optimizer_name)


    train_task = training.TrainTask(
        labeled_data=train_generator(batch_size=batch_size, shuffle=True),
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=optimizer,
        n_steps_per_checkpoint=10,
    )

    eval_task = training.EvalTask(
        labeled_data=val_generator(batch_size=batch_size, shuffle=True),
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )

    model = cl.classifier(len(Vocab))

    print("####### CHECKPOINT 4 ########")

    steps = 100
    training_loop = u.train_model(model, train_task, eval_task, steps, output_dir)
    training_loop.save_checkpoint('CHECKP_' + run.info.run_id)

    print("####### CHECKPOINT 5 ########")

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
    print(accuracy)
    print(f'The accuracy of your model on the validation set is {accuracy:.4f}', )

    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", steps)
    mlflow.log_param("training_size", len(train_x))
    mlflow.log_param("validation_size", len(val_x))
    mlflow.log_metric("val_accuracy", float(accuracy))
    mlflow.log_artifacts("../models")
    #mlflow.log_metric("train_loss", train_loss)
    #mlflow.log_metric("train_accuracy", train_acc)
    #mlflow.log_metric("val_loss", val_loss)
    #mlflow.log_artifacts("./model")

    mlflow.end_run()


