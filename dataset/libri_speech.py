import os
import random

from dataset.dataloader import DataLoader
from settings.constants import SAMPLE_RATE

import numpy as np
import librosa as lr


class LibriSpeech(DataLoader):

    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__(sample_rate)

    def load(self,
             path,
             speakers,
             samples,
             sec_per_sample,
             logging=True):
        data = dict()
        clients = os.listdir(path)
        assert speakers <= len(clients), f'{speakers} too much speakers for dataset'
        audio_samples = samples * sec_per_sample * self.sr
        clients_chosen = 0
        iterations = 0
        while clients_chosen < speakers:
            assert len(clients) != 0, f'{samples} samples, {sec_per_sample} sec per sample - too much for dataset'
            random_client = random.choice(clients)
            if logging:
                print(f'Iteration {iterations}: chosen client {random_client}')
            clients.remove(random_client)
            random_client_audios = []
            for directory in os.listdir(os.path.join(path, random_client)):
                full_path = os.path.join(path, random_client, directory)
                random_client_audios.extend([os.path.join(full_path, audio_path)
                                             for audio_path in os.listdir(full_path)])
            audio = np.array([])
            for audio_path in random_client_audios:
                audio = np.concatenate([audio,
                                        lr.load(
                                            path=audio_path,
                                            sr=self.sr,
                                            mono=True
                                        )[0]])
                if audio.shape[0] >= audio_samples:
                    break
            iterations += 1
            if audio.shape[0] < audio_samples:
                if logging:
                    print(f'Iteration {iterations - 1}: client {random_client} has not enough audio')
                continue
            else:
                clients_chosen += 1
                data[str(clients_chosen)] = np.array([
                    audio[self.sr * sec_per_sample * i:self.sr * sec_per_sample * (i + 1)]
                    for i in range(audio.shape[0] // (sec_per_sample * self.sr))
                ])
                if logging:
                    print(f'Iteration {iterations - 1}: chosen client {random_client}: audio prepared! '
                          f'Client {clients_chosen} is ready!')
        return data
