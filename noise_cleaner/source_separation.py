from models.source_separation.source_separation.synthesize import run


class SourceSeparation:

    def __init__(self, sr, model_name, pretrained_path, lowpass_freq: int = 0):
        self.model_name = model_name
        self.sr = sr
        self.pretrained_path = pretrained_path
        self.lowpass_freq = lowpass_freq

    def clean(self, audio):
        return run(audio,
                   model_name=self.model_name,
                   pretrained_path=self.pretrained_path,
                   lowpass_freq=self.lowpass_freq)
