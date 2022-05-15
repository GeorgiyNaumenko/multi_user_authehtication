import random
import pickle


from settings.constants import DTLN_NORM_40H, LIBRISPEECH_MIXED_DIR, SAMPLE_RATE
from separators.espnet_separator import ESPNET
from noise_cleaner.dtln import DTLN
from authenticator.settings.configs import sample_rate_config, fuzzy_extractor_config, files_config_ls, \
    commonvoice_config, librispeech_config
from dataset.common_voice import CommonVoice
from dataset.libri_speech import LibriSpeech
from dataset.dataloader import DataLoader
from authenticator.model import Authenticator
from feature_extractor.feature_extractor import FeatureExtractor
from fuzzy_extractor.fuzzy_extractor import FuzzyExtractor
from binarizer.binarizer import Binarizer


def prepare_commonvoice(sample_rate=None,
                        source_dir=None,
                        speakers=None,
                        samples=None,
                        sec_per_sample=None,
                        tsv=None,
                        logging=None,
                        target_dir=None
                        ):
    commonvoice = CommonVoice(sample_rate)
    data = commonvoice.load(source_dir,
                            speakers,
                            samples,
                            sec_per_sample,
                            tsv,
                            logging)
    commonvoice.save_data_grouped(data, target_dir)


def prepare_librispeech(sample_rate=None,
                        source_dir=None,
                        speakers=None,
                        samples=None,
                        sec_per_sample=None,
                        logging=None,
                        target_dir=None
                        ):
    librispeech = LibriSpeech(sample_rate)
    data = librispeech.load(source_dir,
                            speakers,
                            samples,
                            sec_per_sample,
                            logging)
    librispeech.save_data_grouped(data, target_dir)


def experiment(diarizator,
               denoiser,
               feature_extractor,
               binarizer,
               fuzzy_extractor,
               dataset_path,
               fuzzy_extractor_config,
               sample_rate_config,
               files_config,
               nums_for_reproduce=5):
    fe = fuzzy_extractor(feature_extractor, binarizer, fuzzy_extractor_config)

    print('Creating authenticator...')
    model = Authenticator(SAMPLE_RATE, diarizator, denoiser, fe)
    print('Authenticator is done.')

    dataloader = DataLoader(sample_rate=SAMPLE_RATE)

    print('Loading data...')
    dataset = dataloader.load_data_grouped(dataset_path)
    print('Data is done.')

    def compare_keys(keys1, keys2):
        return set(keys1) == set(keys2)

    # FAR
    far = dict()
    for pair, audios in dataset.items():

        spk1, spk2 = pair.split('_')

        print(f'Start to calculate FAR for speakers {spk1}, {spk2}')

        key_gen_audio = random.choice(audios)
        print('Keys generation...')
        keys, hints = model.generate_keys(key_gen_audio,
                                          sample_rate_config['diarize_sr'],
                                          sample_rate_config['denoise_sr']
                                          )
        print(f'Keys for pair {spk1}, {spk2} are generated: {keys}')

        far[pair] = {'all': 0,
                     'accepted': 0}

        for other_pair, other_audios in dataset.items():

            other_spk1, other_spk2 = other_pair.split('_')

            if pair != other_pair:

                print(f'Comparing speakers {spk1}, {spk2} to {other_spk1}, {other_spk2}...')

                for key_rep_audio in random.sample(list(other_audios), nums_for_reproduce):

                    print(f'Reproducing keys...')
                    reproduced_keys = model.restore_keys(key_rep_audio,
                                                         hints,
                                                         sample_rate_config['diarize_sr'],
                                                         sample_rate_config['denoise_sr']
                                                         )
                    print(f'For speakers {other_spk1}, {other_spk2} reproduced {reproduced_keys}')

                    far[pair]['all'] += 1
                    if compare_keys(keys, reproduced_keys):
                        print('Other pair was mistakenly accepted :(')
                        far[pair]['accepted'] += 1
                    else:
                        print('Other pair was correctly rejected :)')

    # FRR
    frr = dict()
    for pair, audios in dataset.items():

        spk1, spk2 = pair.split('_')

        print(f'Start to calculate FAR for speakers {spk1}, {spk2}')

        idx = random.randint(0, len(audios) - 1)
        key_gen_audio = audios[idx]
        print(f'Keys generation...')
        keys, hints = model.generate_keys(key_gen_audio,
                                          sample_rate_config['diarize_sr'],
                                          sample_rate_config['denoise_sr']
                                          )
        print(f'Keys for pair {spk1}, {spk2} are generated: {keys}')

        frr[pair] = {'all': 0,
                     'rejected': 0}

        for i in range(len(audios)):
            if i != idx:
                key_rep_audio = audios[i]
                print('Reproducing keys...')
                reproduced_keys = model.restore_keys(key_rep_audio,
                                                     hints,
                                                     sample_rate_config['diarize_sr'],
                                                     sample_rate_config['denoise_sr']
                                                     )
                print(f'Keys for pair {spk1}, {spk2} are reproduced: {reproduced_keys}')

                frr[pair]['all'] += 1
                if not compare_keys(keys, reproduced_keys):
                    frr[pair]['rejected'] += 1
                    print('New sample was mistakenly rejected :(')
                else:
                    print('New sample was correctly accepted :)')

    print('Saving FAR...')
    far_path = files_config['far_path'].format(fuzzy_extractor_config['bytes'], fuzzy_extractor_config['error_bytes'])
    with open(far_path, 'wb') as f:
        try:
            pickle.dump(far, f)
            print('FAR successfully saved!')
        except Exception as e:
            print(f'Error while saving FAR: {e}')

    print('Saving FRR...')
    frr_path = files_config['frr_path'].format(fuzzy_extractor_config['bytes'], fuzzy_extractor_config['error_bytes'])
    with open(frr_path, 'wb') as f:
        try:
            pickle.dump(frr, f)
            print('FRR successfully saved!')
        except Exception as e:
            print(f'Error while saving FRR: {e}')


def run_experiment():
    diarazator = ESPNET(sample_rate=SAMPLE_RATE)
    denoiser = DTLN()
    denoiser.load_model(DTLN_NORM_40H)
    feature_extractor = FeatureExtractor()
    binarizer = Binarizer()

    experiment(diarazator,
               denoiser,
               feature_extractor,
               binarizer,
               FuzzyExtractor,
               LIBRISPEECH_MIXED_DIR,
               fuzzy_extractor_config,
               sample_rate_config,
               files_config_ls,
               nums_for_reproduce=2
               )


if __name__ == '__main__':
    # prepare_commonvoice(**commonvoice_config)
    # prepare_librispeech(**librispeech_config)
    run_experiment()
