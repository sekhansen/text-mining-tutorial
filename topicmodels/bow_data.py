import codecs
from os import path

with codecs.open(path.join(path.dirname(__file__), 'dict_pos_mp.txt'),
                 'r', 'utf-8') as f:
    pos_dict = set(f.read().splitlines())
with codecs.open(path.join(path.dirname(__file__), 'dict_neg_mp.txt'),
                 'r', 'utf-8') as f:
    neg_dict = set(f.read().splitlines())
with codecs.open(path.join(path.dirname(__file__), 'dict_uncertainty.txt'),
                 'r', 'utf-8') as f:
    uncertain_dict = set(f.read().splitlines())
