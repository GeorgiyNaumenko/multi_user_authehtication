import os

import librosa as lr
import numpy as np
import soundfile as sf

from settings.constants import SAMPLE_RATE
from utils.files_handler import folder_state, create_folder


class DataLoader:

    def __init__(self, sample_rate=SAMPLE_RATE):
        """

        :param sample_rate:
        """
        self.sr = sample_rate

    def load_from_folder(self, folder_path):
        """

        :param folder_path:
        :return:
        """
        mixed_data = []
        if folder_state(folder_path):
            filenames = os.listdir(folder_path)
            for filename in filenames:
                y, sr = lr.load(filename, sr=self.sr, mono=True)
                filenames.append(y)
        return mixed_data

    def save_data_to_folder(self, data, folder_path):
        """

        :param data:
        :param folder_path:
        :return:
        """
        create_folder(folder_path)
        for i, file in enumerate(data):
            wav_path = f"{i}.wav"
            sf.write(os.path.join(folder_path, wav_path), file, samplerate=self.sr)

    def load_data_grouped(self, path):
        data = dict()
        for i, subdirectory in enumerate(os.listdir(path)):
            current_data = []
            current_path = os.path.join(path, subdirectory)
            for filename in os.listdir(current_path):
                current_data.append(lr.load(
                    os.path.join(current_path, filename),
                    sr=self.sr,
                    mono=True
                )[0])
            data[subdirectory] = np.array(current_data)
        return data

    def save_data_grouped(self, data, folder_path):
        create_folder(folder_path)
        for subdirectory in data.keys():
            self.save_data_to_folder(data[subdirectory], os.path.join(folder_path, subdirectory))

    @staticmethod
    def mix_pair_audios(audio1, audio2):
        energy1 = np.sum(audio1**2)
        energy2 = np.sum(audio2**2)
        factor = np.sqrt(energy1 / (energy1 + energy2))
        return audio1 * (1 - factor) + audio2 * factor

    def mix_dataset(self, data, logging=True):
        dataset = dict()
        for client1 in data.keys():
            for client2 in data.keys():
                client1_id = int(client1)
                client2_id = int(client2)
                if client1_id > client2_id:
                    name = str(client1) + '_' + str(client2)
                else:
                    name = str(client2) + '_' + str(client1)
                if client1 != client2 and name not in dataset.keys():
                    if logging:
                        print(f'combining {client1} and {client2}....')
                    dataset[name] = np.array(
                        [self.mix_pair_audios(audio1, audio2) for audio1, audio2 in zip(
                            data[client1],
                            data[client2]
                        )]
                    )
        return dataset

