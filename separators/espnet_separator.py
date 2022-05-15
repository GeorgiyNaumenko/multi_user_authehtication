import numpy as np
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.enh_inference import SeparateSpeech

from settings.constants import ESPNET_TAG, SAMPLE_RATE


class ESPNET:

    def __init__(self, model_tag=ESPNET_TAG, sample_rate=SAMPLE_RATE):
        d = ModelDownloader()
        cfg = d.download_and_unpack(model_tag)
        self.model = SeparateSpeech(
            # ESPNET_TAG,
            enh_train_config=cfg["train_config"],
            enh_model_file=cfg["model_file"],
            segment_size=2.4,
            hop_size=0.8,
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=None,
            normalize_output_wav=True,
            device="cpu",
        )
        self.sample_rate = sample_rate

    def separate(self, audio):
        separated = np.array(self.model(audio[None, ...], fs=self.sample_rate))
        return separated.reshape(separated.shape[0], -1)
