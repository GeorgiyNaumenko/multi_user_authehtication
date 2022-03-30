import os

import librosa as lr

from settings.constants import SAMPLE_RATE
from utils.files_hanfler import folder_state, create_folder


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
        pass
