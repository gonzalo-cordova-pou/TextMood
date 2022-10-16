"""
team: TextMood
main_author: @gonzalo-cordova-pou
date: 2020-05-01
description: This script is used to train and evaluate the model choosing the best hyperparameters.
             Results are loaded to comet.ml
"""

import random as rnd
import os
import json
from comet_ml import Experiment
import trax
from trax import fastmath
from trax.supervised import training
from trax import layers as tl
import tensorflow as tf
from codecarbon import EmissionsTracker
import our_model as cl
import prepare as pr
import utils as u
# import trax.layers

NAME = 'MODEL_xlarge_10'
TRAINING_BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 128
STEPS = 500
SIZE = 1600000
TRAINING_PERCENTAGE = 0.7
EMBEDDING_DIM = 256
INNER_DIM = 50
LR = 0.01
OPT = "Adam" # choices are "Adam", "SGD"
OUTPUT_DIR = f"./models/{NAME}/"

experiment = Experiment(
    api_key="0mrbguygGOIO4Gs0ocFddjomE",
    project_name="TEXTMOOD_CO2_TRACKING",
    workspace="textmood",
    log_graph=True,
    auto_param_logging=True,
    auto_metric_logging=True,
    auto_metric_step_rate=1,)

experiment.set_name(NAME)

tracker=EmissionsTracker()
tracker.start()

train_pos, train_neg, val_pos, val_neg, train_x, val_x, train_y, val_y, Vocab = pr.preparation(SIZE, TRAINING_PERCENTAGE)


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
random = fastmath.random # use the fastmath.random module from trax

# ================
# MODEL TRAINING #
# ================

rnd.seed(271)

print("####### CHECKPOINT 3 ########")

# create directory for model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save Vocab to file
with open(OUTPUT_DIR+'Vocab.json', 'w', encoding="utf-8") as fp:
    json.dump(Vocab, fp)

# Choose an optimizer and log it to mlflow
if OPT == "SGD":
    optimizer = trax.optimizers.SGD(learning_rate=LR)
else:
    optimizer = trax.optimizers.Adam(learning_rate=LR)
experiment.log_parameter("optimizer", OPT)


train_task = training.TrainTask(
    labeled_data=train_generator(batch_size=TRAINING_BATCH_SIZE, shuffle=True),
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=optimizer,
    n_steps_per_checkpoint=10,
)

eval_task = training.EvalTask(
    labeled_data=val_generator(batch_size=VALIDATION_BATCH_SIZE, shuffle=True),
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
)

model = cl.classifier(len(Vocab), embedding_dim=EMBEDDING_DIM, inner_dim=INNER_DIM)
experiment.log_parameter("embedding_dim", EMBEDDING_DIM)
experiment.log_parameter("inner_dim", INNER_DIM)

print("####### CHECKPOINT 4 ########")

tf.debugging.set_log_device_placement(True)
with tf.device('/GPU:0'):
    training_loop = u.train_model(model, train_task, eval_task, STEPS, OUTPUT_DIR)
training_loop.save_checkpoint('checkpoint')

print("####### CHECKPOINT 5 ########")

# ================ #
# MODEL EVALUATION IN TEST DATA#
# ================ #

# testing the accuracy of your model: this takes around 20 seconds
model = training_loop.eval_model

accuracy = u.test_model(test_generator(16), model)
print(accuracy)
print(f'The accuracy of your model on the validation set is {accuracy:.4f}', )

experiment.log_parameter("training_batch_size", TRAINING_BATCH_SIZE)
experiment.log_parameter("validation_batch_size", VALIDATION_BATCH_SIZE)
experiment.log_parameter("steps", STEPS)
experiment.log_parameter("training_size", len(train_x))
experiment.log_parameter("validation_size", len(val_x))
experiment.log_parameter("training_percent", TRAINING_PERCENTAGE)
experiment.log_metric("val_accuracy", float(accuracy))

#mlflow.log_metric("train_loss", train_loss)
#mlflow.log_metric("train_accuracy", train_acc)
#mlflow.log_metric("val_loss", val_loss)
#mlflow.log_artifacts("./model")

emissions = tracker.stop()
print("Emissions: ", float(emissions))
experiment.log_metric("emissions", float(emissions))

experiment.end()
