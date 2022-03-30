from settings.constants import SAMPLE_RATE
from dataset.dataloader import DataLoader


class DataSeparator(DataLoader):

    def __init__(self, sample_rate=SAMPLE_RATE, separator=None):
        """

        :param sample_rate:
        :param separator:
        """
        super().__init__(sample_rate)
        self.separator = separator

    def separate_unit(self, audio):
        """

        :param audio:
        :return:
        """
        return self.separator.separate(audio)

    def separate(self, mixed_data):
        """

        :param mixed_data:
        :return:
        """
        separated_data = []
        for audio in mixed_data:
            sep = self.separate_unit(audio)
            separated_data.append(sep)
        return separated_data
