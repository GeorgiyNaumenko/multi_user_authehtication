import numpy as np
import librosa as lr


class Authenticator:

    def __init__(self,
                 sample_rate=None,
                 separator=None,
                 noise_cleaner=None,
                 fuzzy_extractor=None):
        self.sample_rate = sample_rate
        self.noise_cleaner = noise_cleaner
        self.separator = separator
        self.fuzzy_extractor = fuzzy_extractor

    def resample(self, audio, sr_orig, sr_target):
        return lr.resample(audio, sr_orig, sr_target)

    def separate(self, input_audio, sample_rate_required=None):
        if sample_rate_required:
            input_audio = self.resample(input_audio, self.sample_rate, sample_rate_required)
        separated_audios = self.separator.separate(input_audio)
        if sample_rate_required:
            separated_audios = [self.resample(separated, sample_rate_required, self.sample_rate)
                                for separated in separated_audios]
        return np.array(separated_audios)

    def noise_cleaning(self, input_audio, sample_rate_required=None):
        if sample_rate_required:
            input_audio = self.resample(input_audio, self.sample_rate, sample_rate_required)
        cleaned = self.noise_cleaner.clean(input_audio)
        if sample_rate_required:
            cleaned = self.resample(cleaned, sample_rate_required, self.sample_rate)
        return cleaned

    def generate_keys(self, input_audio, sr_sep_req=None, sr_cln_req=None):
        separated = self.separate(input_audio, sr_sep_req)
        cleaned = np.array([self.noise_cleaning(audio, sr_cln_req) for audio in separated])
        keys = []
        hints = []
        for audio in cleaned:
            key, hint = self.fuzzy_extractor.gen(audio)
            keys.append(key.hex())
            hints.append(hint)
        return keys, hints

    def restore_keys(self, input_audio, hints, sr_sep_req=None, sr_cln_req=None):
        separated = self.separate(input_audio, sr_sep_req)
        cleaned = np.array([self.noise_cleaning(audio, sr_cln_req) for audio in separated])
        return [self.fuzzy_extractor.rep(audio, hint).hex() for audio, hint in zip(cleaned, hints)]