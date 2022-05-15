import os

from settings.constants import COMMONVOICE_DIR, COMMONVOICE_CUSTOM_DIR, \
    LIBRISPEECH_DIR, LIBRISPEECH_CUSTOM_DIR, PROJECT_PATH

commonvoice_config = {
    'sample_rate': 16000,
    'speakers': 10,
    'samples': 20,
    'sec_per_sample': 3,
    'tsv': 'validated',
    'source_dir': COMMONVOICE_DIR,
    'logging': True,
    'target_dir': COMMONVOICE_CUSTOM_DIR
}

librispeech_config = {
    'sample_rate': 16000,
    'speakers': 10,
    'samples': 20,
    'sec_per_sample': 3,
    'source_dir': LIBRISPEECH_DIR,
    'logging': True,
    'target_dir': LIBRISPEECH_CUSTOM_DIR
}

far_frr_config = {
    'key_gen_samples_far': 4,
    'key_rep_samples_far': 4,
    'key_gen_samples_frr': 4,
    'key_rep_samples_frr': 2
}

sample_rate_config = {
    'diarize_sr': 8000,
    'denoise_sr': None
}

fuzzy_extractor_config = {
    'bytes': 16,
    'error_bytes': 12
}

files_config_ls = {
    'far_path': os.path.join(PROJECT_PATH, 'metrics', 'far_frr', 'far_ls.pkl'),
    'frr_path': os.path.join(PROJECT_PATH, 'metrics', 'far_frr', 'frr_ls.pkl')
}

files_config_cv = {
    'far_path': os.path.join(PROJECT_PATH, 'metrics', 'librispeech', 'far_frr', 'far_cv_{}_{}.pkl'),
    'frr_path': os.path.join(PROJECT_PATH, 'metrics', 'librispeech', 'far_frr', 'frr_cv_{}_{}.pkl')
}
