"""Constants used in the Notebooks and tvData.py."""
from os.path import abspath

NN_THRESHOLD: int = 30
SCALE: float = 0.2

DUMP: str = abspath('./data/dump/')

FILES_S4 = [
    'gr_ringstrecke_s4_04',
    'gr_ringstrecke_s4_05',
    'gr_ringstrecke_s4_06',
    'gr_ringstrecke_s4_07',
    'gr_ringstrecke_s4_08',
    'gr_ringstrecke_s4_09',
    'gr_ringstrecke_s4_10',
    'gr_ringstrecke_s4_11',
    'gr_ringstrecke_s4_12',
    'gr_ringstrecke_s4_13',
    'gr_ringstrecke_s4_14',
    ]

# complete s4 runs without any powerloss transmitter
GOOD_FILES_S4 = [
    'gr_ringstrecke_s4_11',
    'gr_ringstrecke_s4_06',
    'gr_ringstrecke_s4_14',
    ]