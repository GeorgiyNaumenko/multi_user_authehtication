from models.fuzzy_extractor import FuzzyExtractor as FE


class FuzzyExtractor():

    def __init__(self, feature_extractor, binarizer, config_dict):
        self.fe = FE(config_dict['bytes'], config_dict['error_bytes'])
        self.feature_extractor = feature_extractor
        self.binarizer = binarizer

    def gen(self, data):
        features_extracted = self.feature_extractor.extract(data)
        features = self.binarizer.binarize(features_extracted)
        return self.fe.generate(features)

    def rep(self, data, hint):
        features_extracted = self.feature_extractor.extract(data)
        features = self.binarizer.binarize(features_extracted)
        return self.fe.reproduce(features, hint)