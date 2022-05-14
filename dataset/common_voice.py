import os
import random

from dataset.dataloader import DataLoader
from settings.constants import SAMPLE_RATE

import pandas as pd
import numpy as np
import librosa as lr


class CommonVoice(DataLoader):

    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__(sample_rate)

    def load(self,
             path,
             speakers,
             samples,
             sec_per_sample,
             tsv,
             logging=True):
        data = dict()
        dataframe = pd.read_csv(os.path.join(path, tsv + '.tsv'), sep="\t")
        dataframe = dataframe[dataframe['client_id'].notna()]
        clients = dataframe['client_id'].unique()
        clients = list(clients)
        assert speakers <= len(clients), f'{speakers} too much speakers for dataset {tsv}'
        audio_samples = samples * sec_per_sample * self.sr
        clients_chosen = 0
        iterations = 0
        while clients_chosen < speakers:
            assert len(clients) != 0, f'{samples} samples, {sec_per_sample} sec per sample - too much for dataset {tsv}'
            random_client = random.choice(clients)
            if logging:
                print(f'Iteration {iterations}: chosen client {random_client}')
            clients.remove(random_client)
            random_clients_df = dataframe[dataframe['client_id'] == random_client]
            random_client_audios = random_clients_df['path']
            audio = np.array([])
            for audio_path in random_client_audios:
                audio = np.concatenate([audio,
                                        lr.load(
                                            path=os.path.join(path, 'clips', audio_path),
                                            sr=self.sr,
                                            mono=True
                                        )[0]])
                if audio.shape[0] >= audio_samples:
                    break
            iterations += 1
            if audio.shape[0] < audio_samples:
                if logging:
                    print(f'Iteration {iterations-1}: client {random_client} has not enough audio')
                continue
            else:
                clients_chosen += 1
                data[str(clients_chosen)] = np.array([
                    audio[self.sr * sec_per_sample * i:self.sr * sec_per_sample * (i + 1)]
                    for i in range(audio.shape[0] // (sec_per_sample * self.sr))
                ])
                if logging:
                    print(f'Iteration {iterations-1}: chosen client {random_client}: audio prepared! '
                          f'Client {clients_chosen} is ready!')
        return data
