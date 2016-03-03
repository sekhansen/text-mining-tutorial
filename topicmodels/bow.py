"""
(c) 2015, Stephen Hansen, stephen.hansen@upf.edu
"""

from __future__ import division

import collections,itertools
import numpy as np

class BOW():

	"""
	Form document-term matrix, and perform related operations.
	"""

	def __init__(self,docs):

		self.D = len(docs)
		self.docs = docs

		doc_list = list(itertools.chain(*docs))
		self.token_key = {}
		for i,v in enumerate(set(doc_list)): self.token_key[v] = i
		self.V = len(self.token_key)

		self.bow = np.zeros((self.D,self.V),dtype=np.int)

		for d,doc in enumerate(docs):
			temp = collections.Counter(doc)
			for v in temp.keys(): self.bow[d,self.token_key[v]] = temp[v]



	def tf_idf(self):

		idf = np.log(self.bow.shape[0]/np.where(self.bow > 0,1,0).sum(axis=0))

		tf = np.log(self.bow)
		tf[tf==-np.inf]=-1
		tf = tf + 1

		return tf * idf