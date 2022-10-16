import json
from pyexpat import model
import our_model as cl
import utils as u
from pathlib import Path
from trax import fastmath

# use the numpy module from trax
np = fastmath.numpy

class TextMoodModel():
    '''Check model'''
    def __init__(self, model_name):      
        self.name = model_name
        self.directory = f"../models/{model_name}/"

        print("Loading vocab...")
        try:
            with open(self.directory+'vocab.json', 'r') as f_p:
                self.vocab = json.load(f_p)
        except:
            raise Exception("vocab file not found. Please check the model directory.")      
        print("Loading model architecture...")
        try:
            self.architecture = cl.classifier(len(self.vocab))
        except:
            raise Exception("Model architecture returned an error. Please check the model architecture in our_model.py.")

        print("Success!")

    def initialize_model(self):
        '''Inicialization of the model'''

        try:
            self.architecture.init_from_file(self.directory+ 'checkpoint.pkl.gz', weights_only=True)
        except:
            raise Exception("Model weights in ({}) not found. Please check the model directory.".format(self.directory+ 'checkpoint.pkl.gz'))

        print("Model initialized.")

    def predict(self, sentence):
        """
        Predict the sentiment of a sentence.
        """

        if self.architecture is None:
            raise Exception("Model not initialized. Please run self.initialize_model() first.")

        inputs = np.array(u.tweet2tensor(sentence, vocab_dict=self.vocab))

        # Batch size 1, add dimension for batch, to work with the model
        inputs = inputs[None, :]

        # predict with the model
        preds_probs = self.architecture(inputs)

        # Turn probabilities into categories
        preds = int(preds_probs[0, 1] > preds_probs[0, 0])

        return preds

    def predict_batch(self, sentences):
        """
        Predict the sentiment of a batch of sentences.
        """

        if self.architecture is None:
            raise Exception("Model not initialized. Please run self.initialize_model() first.")

        return [self.predict(sentence) for sentence in sentences]

def get_model_names():
    """
    Get the names of all the models in the models directory.
    """
    return [x.name for x in Path('../models/').iterdir() if x.is_dir()]
