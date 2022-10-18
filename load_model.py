import boto3
import os

def load_model(destination):

    DIRNAME = os.path.dirname(os.path.abspath(__file__))

    AWS_ACCESS_KEY_ID = 'AKIAQ6I2MOXSLD4G2YGT'
    AWS_SECRET_ACCESS_KEY = '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'
    session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    s3 = session.resource('s3', region_name='eu-west-3')
    print("Downloading model weights...")
    s3.Bucket('tracktrend.models').download_file('checkpoint.weights.npy.gz', os.path.join(DIRNAME, destination+'/checkpoint.weights.npy.gz'))
    print("Downloading model parameters...")
    s3.Bucket('tracktrend.models').download_file('checkpoint.pkl.gz', os.path.join(DIRNAME, destination+'/checkpoint.pkl.gz'))
    print("Downloading model vocabulary...")
    s3.Bucket('tracktrend.models').download_file('Vocab.json', os.path.join(DIRNAME, destination+'/Vocab.json'))