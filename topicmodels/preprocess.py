"""
(c) 2015, Stephen Hansen, stephen.hansen@upf.edu
"""

from __future__ import division

import codecs,collections,itertools,re
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer

import preprocess_data # contains stopwords and contractions

pattern = re.compile('\W',re.UNICODE)

def my_tokenize(text): return re.split(pattern,text)

class RawDocs():
    
	def __init__(self, doc_data, stopwords = set(), contraction_split = True):

		"""
		doc_data: (1) text file with each document on new line, or (2) Python iterable of strings.
			Strings should have utf-8 encoded characters.
		stopwords: 'long' is longer list of stopwords, 'short' is shorter list of stopwords.
		contraction_split: whether to split contractions into constituent words.  If not, remove all apostrophes.
		"""

		if isinstance(doc_data,str):
			try: 
				with codecs.open(doc_data,'r','utf-8') as f: raw = f.read()
			except UnicodeDecodeError: print "File does not have utf-8 encoding"
			self.docs = raw.splitlines()
		elif isinstance(doc_data, collections.Iterable):
			try: self.docs = [s.encode('utf-8').decode('utf-8') for s in doc_data]
			except UnicodeDecodeError: print "At least one string does not have utf-8 encoding"
		else:
			raise ValueError("Either iterable of strings or file must be passed to RawDocs")

		self.docs = [s.lower() for s in self.docs]

		if stopwords == 'long': self.stopwords = preprocess_data.stp_long
		elif stopwords == 'short': self.stopwords = preprocess_data.stp_short

		if contraction_split:
			for k,v in preprocess_data.contractions.iteritems():
				self.docs = map(lambda x: re.sub(k,v,x),self.docs)
		else:
			self.docs = map(lambda x: re.sub(u'[\u2019\']', '', x), self.docs)

		self.N = len(self.docs)
		self.tokens = map(wordpunct_tokenize,self.docs)
		

	def phrase_replace(self,replace_dict):

		"""
		Replace phrases with single token, mapping defined in replace_dict
		"""

		def r(tokens):
			text = ' ' + ' '.join(tokens)
			for k,v in replace_dict.iteritems():
				text = text.replace(" " + k + " "," " + v + " ")
			return text.split()
		self.stems = map(r,self.stems)

		
	def token_clean(self,length,numbers=True):

		"""
		Strip out non-alphanumeric tokens.
		length: remove tokens of length "length" or less.
		numbers: strip out non-alpha tokens.
		"""

		def clean1(tokens): return [t for t in tokens if t.isalpha() == 1 and len(t) > length]
		def clean2(tokens): return [t for t in tokens if t.isalnum() == 1 and len(t) > length]

		if numbers: self.tokens = map(clean1,self.tokens)
		else: self.tokens = map(clean2,self.tokens)


	def stem(self):

		"""
		Stem tokens with Porter Stemmer.
		"""

		def s(tokens): return [PorterStemmer().stem(t) for t in tokens]
		self.stems = map(s,self.tokens)


	def bigram(self,items):

		"""
		generate bigrams of either items = "tokens" or "stems"
		"""

		def bigram_join(tok_list): 
			text = nltk.bigrams(tok_list)
			return map(lambda x: x[0] + '.' + x[1],text)

		if items=="tokens":self.bigrams = map(bigram_join,self.tokens)
		elif items=="stems":self.bigrams = map(bigram_join,self.stems)
		else: raise ValueError("Items must be either \'tokens\' or \'stems\'.")
		

	def stopword_remove(self,items,threshold=False):

		"""
		Remove stopwords from either tokens (items = "tokens") or stems (items = "stems")
		"""

		def remove(tokens): return [t for t in tokens if t not in self.stopwords]
	
		if items == 'tokens': self.tokens = map(remove,self.tokens)
		elif items == 'stems': self.stems = map(remove,self.stems)
		else: raise ValueError("Items must be either \'tokens\' or \'stems\'.")

		
	def term_rank(self,items,print_output=True):

		"""
		Calculate corpus-level df and tf-idf scores on either tokens (items = "tokens") or stems (items = "stems").
		Print to file if print_output = True.
		"""
	
		if items == 'stems': v = self.stems
		elif items == 'tokens': v = self.tokens
	
		agg = itertools.chain(*v)
		counts = collections.Counter(agg)
		
		v_unique = map(lambda x: set(x),v)
		agg_d = itertools.chain(*v_unique)
		counts_d = collections.Counter(agg_d)
		
		unique_tokens = set(itertools.chain(*v))

		def tf_idf_compute(t): return (1 + np.log(counts[t]))*np.log(self.N/counts_d[t])

		unsorted_df = [counts[t] for t in unique_tokens]
		unsorted_tf_idf = [tf_idf_compute(t) for t in unique_tokens]

		self.df_ranking = sorted(zip(unique_tokens,unsorted_df),key=lambda x: x[1],reverse=True)
		self.tfidf_ranking = sorted(zip(unique_tokens,unsorted_tf_idf),key=lambda x: x[1],reverse=True)

		if print_output:
			with codecs.open('df_ranking.csv','w','utf-8') as f:
				for p in self.df_ranking: f.write("%s,%d\n" % (p[0],p[1]))
			with codecs.open('tfidf_ranking.csv','w','utf-8') as f:
				for p in self.tfidf_ranking: f.write("%s,%f\n" % (p[0],p[1]))

		else: raise ValueError("Items must be either \'tokens\' or \'stems\'.")


	def rank_remove(self,rank,items,cutoff):

		"""
		remove tokens or stems (specified in items) based on rank's (df or tfidf) value being less than cutoff
		to remove all words with rank R or less, specify cutoff = self.xxx_ranking[R][1]
		"""

		def remove(tokens): return [t for t in tokens if t not in to_remove]

		if rank == "df": to_remove = set([t[0] for t in self.df_ranking if t[1] <= cutoff])
		elif rank == "tfidf": to_remove = set([t[0] for t in self.tfidf_ranking if t[1] <= cutoff])
		else: raise ValueError("Rank must be either \'df\' or \'tfidf\'.")

		if items == 'tokens': self.tokens = map(remove,self.tokens)
		elif items == 'stems': self.stems = map(remove,self.stems)
		else: raise ValueError("Items must be either \'tokens\' or \'stems\'.")

