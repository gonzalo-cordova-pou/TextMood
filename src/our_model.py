from pyexpat import model
import trax_models
from trax import layers as tl

def classifier(vocab_size, embedding_dim=256, output_dim=2, model='train'):
    '''Classifies user mood regarding their tweets'''
    #Create embedding layer
    embed_layer = tl.Embedding(
    vocab_size=vocab_size, # Size of the vocabulary
    d_feature=embedding_dim)  # Embedding dimension
    # Create a mean layer, to create an "average" word embedding
    mean_layer = tl.Mean(axis=1)
    # Create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units = output_dim)
    # Create the log softmax layer (no parameters needed)
    log_softmax_layer = tl.LogSoftmax()
    # Use tl.Serial to combine all layer
    # and create the classifier
    # of type trax.layers.combinators.Serial
    model = tl.Serial(
    embed_layer, # embedding layer
    mean_layer, # mean layer
    dense_output_layer, # dense output layer
    log_softmax_layer  # log softmax layer
    )
    return model