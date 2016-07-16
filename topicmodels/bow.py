"""
(c) 2015, Stephen Hansen, stephen.hansen@upf.edu
"""

from __future__ import division

import collections
import itertools
import numpy as np

import bow_data  # contains dictionaries of interest
from nltk import PorterStemmer


class BOW():

    """
    Form document-term matrix, and perform related operations.
    """

    def __init__(self, docs):

        self.D = len(docs)
        self.docs = docs

        doc_list = list(itertools.chain(*docs))
        self.token_key = {}
        for i, v in enumerate(set(doc_list)):
            self.token_key[v] = i
        self.V = len(self.token_key)

        self.bow = np.zeros((self.D, self.V), dtype=np.int)

        for d, doc in enumerate(docs):
            temp = collections.Counter(doc)
            for v in temp.keys():
                self.bow[d, self.token_key[v]] = temp[v]

    def tf_idf(self):

        idf = np.log(self.bow.shape[0]/np.where(self.bow > 0, 1, 0).
                     sum(axis=0))

        tf = np.log(self.bow)
        tf[tf == -np.inf] = -1
        tf = tf + 1

        self.tf_idf_mat = tf * idf

    def dict_count(self, dictionary):

        indices = [self.token_key[v] for v in dictionary]

        return self.bow[:, indices].sum(axis=1)

    def pos_count(self, items):

        if items == "tokens":
                return self.dict_count(bow_data.pos_dict)

        elif items == "stems":

            pos_stems = [PorterStemmer().stem(t) for t in
                         bow_data.pos_dict.intersection(set(self.token_key.
                                                            keys()))]
            return self.dict_count(pos_stems)

        else:
            raise ValueError("Items must be either \'tokens\' or \'stems\'.")

    def neg_count(self, items):

        if items == "tokens":
            return self.dict_count(bow_data.neg_dict)

        elif items == "stems":

            neg_stems = [PorterStemmer().stem(t) for t in
                         bow_data.neg_dict.intersection(set(self.token_key.
                                                            keys()))]
            return self.dict_count(neg_stems)

        else:
            raise ValueError("Items must be either \'tokens\' or \'stems\'.")

    def uncertain_count(self, items):

        if items == "tokens":
            return self.dict_count(bow_data.uncertain_dict)

        elif items == "stems":

            uncertain_stems = [PorterStemmer().stem(t) for t in
                               bow_data.uncertain_dict.intersection(
                                   set(self.token_key.keys())
                               )]
            return self.dict_count(uncertain_stems)

        else:
            raise ValueError("Items must be either \'tokens\' or \'stems\'.")
