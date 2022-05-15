import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class FeatureExtractor:

    def __init__(self):
        print('Downloading VGGish...')
        self.model = hub.load('https://tfhub.dev/google/vggish/1')
        print('Downloading VGGish is completed.')

    def extract(self, audio):
        return self.model(audio).numpy()
