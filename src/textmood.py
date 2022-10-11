import json
import our_model as cl
import utils as u
from trax import fastmath

# use the numpy module from trax
np = fastmath.numpy

class TextMoodModel():
    
    def __init__(self, model_name):

        
        self.name = model_name
        self.directory = './models/{}/'.format(model_name)
        
        print("Loading vocab...")
        try:
            with open(self.directory+'Vocab.json', 'r') as fp:
                self.Vocab = json.load(fp)
        except:
            raise Exception("Vocab file not found. Please check the model directory.")
        
        print("Loading model architecture...")
        try:
            self.architecture = cl.classifier(len(self.Vocab))
        except:
            raise Exception("Model architecture returned an error. Please check the model architecture in our_model.py.")

        print("Success!")
        
    def initialize_model(self):

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
        
        inputs = np.array(u.tweet2tensor(sentence, vocab_dict=self.Vocab))

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