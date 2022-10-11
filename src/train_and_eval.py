import random as rnd
import os
import trax
from trax.supervised import training
import json
import mlflow
import our_model as cl
import prepare as pr
from mlflow import pyfunc
from trax import fastmath
import tensorflow as tf
import utils as u
# import trax.layers
from trax import layers as tl

NAME = 'model_0'
TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
STEPS = 200
SIZE = 20000 #1000000
TRAINING_PERCENTAGE = 0.8
# pylint: disable=consider-using-f-string
OUTPUT_DIR = './models/{}/'.format(NAME)


[train_pos, train_neg, val_pos, val_neg, train_x,
val_x, train_y, val_y, Vocab] = pr.preparation(SIZE, TRAINING_PERCENTAGE)


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


def train_generator(batch_size, shuffle = False):
    '''Create the training data generator'''
    return u.data_generator(train_pos, train_neg, batch_size, True, Vocab, shuffle)

def val_generator(batch_size, shuffle = False):
    '''Create the validation data generator'''
    return u.data_generator(val_pos, val_neg, batch_size, True, Vocab, shuffle)

def test_generator(batch_size, shuffle = False):
    '''Create the validation data generator'''
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

# ================
# MODEL TRAINING #
# ================

tmp_model = cl.classifier(len(Vocab))

rnd.seed(271)

print("####### CHECKPOINT 3 ########")

mlflow.tensorflow.autolog()

with mlflow.start_run(run_name=NAME) as run:

    # create directory for model
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save Vocab to file
    with open(OUTPUT_DIR+'Vocab.json', 'w', encoding="utf-8") as fp:
        json.dump(Vocab, fp)

    # Choose an optimizer and log it to mlflow
    LR = 0.01
    OPT = "Adam" # choices are "Adam", "SGD"
    if OPT == "SGD":
        optimizer = trax.optimizers.SGD(learning_rate=LR)
    else:
        optimizer = trax.optimizers.Adam(learning_rate=LR)
    mlflow.log_param("optimizer", OPT)


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

    model = cl.classifier(len(Vocab))

    print("####### CHECKPOINT 4 ########")

    training_loop = u.train_model(model, train_task, eval_task, STEPS, OUTPUT_DIR)
    training_loop.save_checkpoint('checkpoint')

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

    mlflow.log_param("TRAINING_BATCH_SIZE", TRAINING_BATCH_SIZE)
    mlflow.log_param("VALIDATION_BATCH_SIZE", VALIDATION_BATCH_SIZE)
    mlflow.log_param("STEPS", STEPS)
    mlflow.log_param("training_size", len(train_x))
    mlflow.log_param("validation_size", len(val_x))
    mlflow.log_param("training_percent", TRAINING_PERCENTAGE)
    mlflow.log_metric("val_accuracy", float(accuracy))
    mlflow.log_artifacts("./models")
    #mlflow.log_metric("train_loss", train_loss)
    #mlflow.log_metric("train_accuracy", train_acc)
    #mlflow.log_metric("val_loss", val_loss)
    #mlflow.log_artifacts("./model")

    mlflow.end_run()


