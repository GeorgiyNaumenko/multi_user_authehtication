import os

import librosa as lr
import numpy as np

from models.DTLN.DTLN_model import DTLN_model


class DTLN:

    def __init__(self):
        self.sample_rate = 16000
        self.model = None

    def load_model(self, model):
        self.model = DTLN_model()
        self.model.build_DTLN_model(norm_stft=model['norm_stft'])
        self.model.model.load_weights(model['path'])

    def clean(self, audio):
        # get length of file
        len_orig = len(audio)
        # pad audio
        zero_pad = np.zeros(384)
        in_data = np.concatenate((zero_pad, audio, zero_pad), axis=0)
        # predict audio with the model
        predicted = self.model.model.predict_on_batch(
            np.expand_dims(in_data, axis=0).astype(np.float32))
        # squeeze the batch dimension away
        predicted_speech = np.squeeze(predicted)
        predicted_speech = predicted_speech[384:384 + len_orig]
        return predicted_speech
