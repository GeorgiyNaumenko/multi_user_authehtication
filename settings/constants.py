import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

DATA_DIR = os.path.join(PROJECT_PATH, 'data')
MIXED_DIR = os.path.join(DATA_DIR, 'mix')
NOISY_DIR = os.path.join(DATA_DIR, 'noisy')
CLEANED_DIR = os.path.join(DATA_DIR, 'clean')
COMMONVOICE_DIR = os.path.join(DATA_DIR, 'cv-corpus-9.0-2022-04-27', 'uk')
COMMONVOICE_CUSTOM_DIR = os.path.join(DATA_DIR, 'CommonVoiceCustom')
COMMONVOICE_MIXED_DIR = os.path.join(DATA_DIR, 'CommonVoiceMixed')
LIBRISPEECH_DIR = os.path.join(DATA_DIR, 'LibriSpeech', 'dev-other')
LIBRISPEECH_CUSTOM_DIR = os.path.join(DATA_DIR, 'LibriSpeechCustom')
LIBRISPEECH_MIXED_DIR = os.path.join(DATA_DIR, 'LibriSpeechMixed')


SAMPLE_RATE = 16000

ESPNET_TAG = "lichenda/Chenda_Li_wsj0_2mix_enh_dprnn_tasnet"

SOURCE_SEP_PRTRN_PATH = os.path.join(PROJECT_PATH, 'models', 'source_separation', 'pretrained')
REFINE_SPECTROGRAM_UNET = {'name': 'refine_unet_larger',
                           'path': os.path.join(SOURCE_SEP_PRTRN_PATH, 'RefineSpectrogramUnet.best.chkpt')}
SECOND_VOICE_BANK = {'name': 'refine_unet_base',
                     'path': os.path.join(SOURCE_SEP_PRTRN_PATH, 'second_voice_bank.best.chkpt')}


MODEL_DTLN_PRETRAINED = os.path.join(MODELS_PATH, 'DTLN', 'pretrained_model')
DTLN_NORM_40H = {'path': os.path.join(MODEL_DTLN_PRETRAINED, 'DTLN_norm_40h.h5'),
                 'norm_stft': True}
DTLN_NORM_500H = {'path': os.path.join(MODEL_DTLN_PRETRAINED, 'DTLN_norm_500h.h5'),
                  'norm_stft': True}
DTLN_PRETR = {'path': os.path.join(MODEL_DTLN_PRETRAINED, 'model.h5'),
              'norm_stft': False}


