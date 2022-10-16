import tensorflow as tf
import utils as u
import prepare as pr
import utils as u
from trax import layers as tl
from trax.supervised import training
import numpy as np
import our_model as cl
import json

# Choose the model version
CHOOSE_VERSION = 'model_0'

# Load Vocab
print("Loadin vocab...")
with open('Vocab.json', 'r') as fp:
    Vocab = json.load(fp)

def predict(sentence):
    ''' Predict on your own sentnece'''
    inputs = np.array(u.tweet2tensor(sentence, vocab_dict=Vocab))
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

print("Loadin model architecture...")
# Load the model architecture
model = cl.classifier(len(Vocab))

OUTPUT_DIR = './models/{}/'.format(CHOOSE_VERSION)

# Initialize using pre-trained weights
print("Initializing model...")
model.init_from_file(OUTPUT_DIR + 'checkpoint.pkl.gz'.format(CHOOSE_VERSION), weights_only=True)

# try a positive SENTENCE
print("Testing model...")
# pylint: disable=line-too-long
SENTENCE = "It's such a nice day, think i'll be taking Sid to Ramsgate fish and chips for lunch at Peter's fish factory and then the beach maybe"
tmp_pred, tmp_sentiment = predict(SENTENCE)
print(f"The sentiment of the SENTENCE \n***\n\"{SENTENCE}\"\n***\nis {tmp_sentiment}.")

print()
# try a negative SENTENCE
SENTENCE = "I hated my day, it was the worst, I'm so sad."
tmp_pred, tmp_sentiment = predict(SENTENCE)
print(f"The sentiment of the SENTENCE \n***\n\"{SENTENCE}\"\n***\nis {tmp_sentiment}.")

# Load the model from SavedModel.
#loaded_model = tf.keras.models.load_model('./keras_model/model_checkpoint/')

# Run the loaded model to verify it returns the same result.
#sentiment_activations = loaded_model(tmp_inputs[None, :])
#print(f'Keras returned sentiment activations: {np.asarray(sentiment_activations)}')
