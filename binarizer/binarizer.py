import numpy as np


class Binarizer:

    def __init__(self, deg=8):
        self.deg=deg

    @staticmethod
    def select_stable(features, num_features_vectors=4, num=16):
        choose = np.min([num_features_vectors, features.shape[0]])
        corr_sum = np.sum(np.corrcoef(features), axis=1)
        idxs = np.argsort(corr_sum)
        idxs = idxs[-choose:]
        features = features[idxs]
        std_array = np.std(features, axis=0)
        idxs_std = np.argsort(std_array)
        idxs_std = idxs_std[-num:]
        features = features[:, idxs_std]
        features = np.mean(features, axis=0)
        return features

    def quantize(self, array, alpha=1):
        array = self.sigmoid(alpha*array)
        return np.round(array*(2**self.deg - 1)).astype(np.uint8)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tostring_num(self, number):
        if not any(isinstance(number, integer) for integer in [np.integer, int]):
            raise Exception('number must be integer (supported int, numpy.integer)')
        if not 0 <= number <= 2**self.deg - 1:
            raise Exception(f'number must be between 0 and {2**self.deg - 1}')
        return chr(number)

    def tostring(self, array):
        return ''.join([self.tostring_num(number) for number in array])

    def binarize(self, features, num_features_vectors=4, bytes_num=16):
        features = self.select_stable(features, num_features_vectors, bytes_num)
        features = self.quantize(features)
        return features
