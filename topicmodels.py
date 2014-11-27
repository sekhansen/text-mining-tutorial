"""
(c) 2014, Stephen Hansen, stephen.hansen@upf.edu

Python module for topic modelling with three classes:
(1) RawDocs: cleans and processes raw text data for input into topic model.
(2) Count: count tokens in processed text
(2) LDA: estimate Latent Dirichlet Allocation topic model via collapsed Gibbs sampling.
(3) Query: use estimated topic model to sample distribution of topics in out-of-sample documents.

"""

from __future__ import division

import codecs,collections,itertools,os,re
import numpy as np
import pandas as pd

from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer

import samplers

contractions = {
	u"ain[\u2019\']t" : u"is not",
	u"aren[\u2019\']t" : u"are not",
	u"can[\u2019\']t" : u"cannot",
	u"could[\u2019\']ve" : u"could have",
	u"couldn[\u2019\']t" : u"could not",
	u"didn[\u2019\']t" : u"did not",
	u"doesn[\u2019\']t" : u"does not",
	u"don[\u2019\']t" : u"do not",
	u"hadn[\u2019\']t" : u"had not",
	u"hasn[\u2019\']t" : u"has not",
	u"haven[\u2019\']t" : u"have not",
	u"he[\u2019\']d" : u"he would",
	u"he[\u2019\']ll" : u"he will",
	u"he[\u2019\']s" : u"he is",
	u"how[\u2019\']d" : u"how did",
	u"how[\u2019\']ll" : u"how will",
	u"how[\u2019\']s" : u"how is",
	u"i[\u2019\']d" : u"i would",
	u"i[\u2019\']ll" : u"i will",
	u"i[\u2019\']m" : u"i am",
	u"i[\u2019\']ve" : u"i have",
	u"isn[\u2019\']t" : u"is not",
	u"it[\u2019\']d" : u"it would",
	u"it[\u2019\']ll" : u"it will",
	u"it[\u2019\']s" : u"it is",
	u"let[\u2019\']s" : u"let us",
	u"ma[\u2019\']am" : u"madam",
	u"might[\u2019\']ve" : u"might have",
	u"must[\u2019\']ve" : u"must have",
	u"needn[\u2019\']t" : u"need not",
	u"o[\u2019\']clock" : u"of the clock",
	u"shan[\u2019\']t" : u"shall not",
	u"she[\u2019\']d" : u"she would",
	u"she[\u2019\']ll" : u"she will",
	u"she[\u2019\']s" : u"she is",
	u"should[\u2019\']ve" : u"should have",
	u"shouldn[\u2019\']t" : u"should not",
	u"that[\u2019\']d" : u"that would",
	u"that[\u2019\']ll" : u"that will",
	u"that[\u2019\']s" : u"that is",
	u"there[\u2019\']d" : u"there would",
	u"there[\u2019\']ll" : u"there will",
	u"there[\u2019\']s" : u"there is",
	u"they[\u2019\']d" : u"they would",
	u"they[\u2019\']ll" : u"they will",
	u"they[\u2019\']re" : u"they are",
	u"they[\u2019\']ve" : u"they have",
	u"wasn[\u2019\']t" : u"was not",
	u"we[\u2019\']d" : u"we would",
	u"we[\u2019\']ll" : u"we will",
	u"we[\u2019\']re" : u"we are",
	u"we[\u2019\']ve" : u"we have",
	u"weren[\u2019\']t" : u"were not",
	u"what[\u2019\']ll" : u"what will",
	u"what[\u2019\']re" : u"what are",
	u"what[\u2019\']s" : u"what is",
	u"when[\u2019\']s" : u"when is",
	u"where[\u2019\']d" : u"where did",
	u"where[\u2019\']s" : u"where is",
	u"where[\u2019\']ve" : u"where have",
	u"who[\u2019\']ll" : u"who will",
	u"who[\u2019\']s" : u"who is",
	u"who[\u2019\']ve" : u"who have",
	u"why[\u2019\']s" : u"why is",
	u"won[\u2019\']t" : u"will not",
	u"would[\u2019\']ve" : u"would have",
	u"wouldn[\u2019\']t" : u"would not",
	u"y[\u2019\']all" : u"you all",
	u"you[\u2019\']d" : u"you would",
	u"you[\u2019\']ll" : u"you will",
	u"you[\u2019\']re" : u"you are",
	u"you[\u2019\']ve" : u"you have"
}

class RawDocs():
    
	def __init__(self, doc_data, stopword_file = False, contraction_split = True):

		"""
		doc_data: (1) text file with each document on new line, or (2) Python iterable of strings.
			Strings should have utf-8 encoded characters.

		stopword_file: list of stopwords to remove (optional)

		contraction_split: whether to split contractions into constituent words.  
					       If not, remove all apostrophes.
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

		if stopword_file:
			with codecs.open(stopword_file,'r','utf-8') as f: raw = f.read()
			self.stopwords = set(raw.splitlines())

		if contraction_split:
			for k,v in contractions.iteritems():
				self.docs = map(lambda x: re.sub(k,v,x),self.docs)
		else:
			self.docs = map(lambda x: re.sub(u'[\u2019\']', '', x), self.docs)

		self.N = len(self.docs)
		self.tokens = map(wordpunct_tokenize,self.docs)

		
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

		
	def stopword_remove(self,items,threshold=False):

		"""
		Remove stopwords from either tokens (items = "tokens") or stems (items = "stems")

		threshold: remove words whose corpus-level tf-idf score is less than or 
		equal to that of the treshold-th ranked item (requires call to tf_idf first to define ranking)
		"""
	
		def remove(tokens): return [t for t in tokens if t not in to_remove]
		
		if items == 'tokens':
			if threshold:
				cutoff = self.tfidf_ranking_tokens[threshold][1]
				stopwords_ranking = [t[0] for t in self.tfidf_ranking_tokens if t[1] <= cutoff]
				to_remove = self.stopwords.union(stopwords_ranking)
			else: to_remove = self.stopwords
			self.tokens = map(remove,self.tokens)

		elif items == 'stems': 
			if threshold:
				cutoff = self.tfidf_ranking_stems[threshold][1]
				stopwords_ranking = [t[0] for t in self.tfidf_ranking_stems if t[1] <= cutoff]
				to_remove = self.stopwords.union(stopwords_ranking)
			else: to_remove = self.stopwords
			self.stems = map(remove,self.stems)

		else: 
			raise ValueError("Items must be either \'tokens\' or \'stems\'.")

		
	def tf_idf(self,items,print_output=True):

		"""
		Calculate corpus-level tf-idf score on either tokens (items = "tokens") or stems (items = "stems").

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

		unsorted_tf_idf = [tf_idf_compute(t) for t in unique_tokens]

		if items == 'tokens':

			self.tfidf_ranking_tokens = sorted(zip(unique_tokens,unsorted_tf_idf),key=lambda x: x[1],reverse=True)

			if print_output:
				with codecs.open('tfidf_ranking_tokens.csv','w','utf-8') as f:
					for p in self.tfidf_ranking_tokens: f.write("%s,%f\n" % (p[0],p[1]))

		elif items == 'stems':

			self.tfidf_ranking_stems = sorted(zip(unique_tokens,unsorted_tf_idf),key=lambda x: x[1],reverse=True)

			if print_output:
				with codecs.open('tfidf_ranking_stems.csv','w','utf-8') as f:
					for p in self.tfidf_ranking_stems: f.write("%s,%f\n" % (p[0],p[1]))

		else: 
			raise ValueError("Items must be either \'tokens\' or \'stems\'.")


class LDA():

	def __init__(self, docs, K):

		"""
		Initialize a topic model for Gibbs sampling.

		docs: list (documents) of lists (string tokens within documents)
		K: number of topics
		"""

		self.K = K
		self.D = len(docs)

		doc_list = list(itertools.chain(*docs))
		self.token_key = {}
		for i,v in enumerate(set(doc_list)): self.token_key[v] = i
		self.V = len(self.token_key)

		self.tokens = np.array([self.token_key[t] for t in doc_list], dtype = np.int)
		self.N = self.tokens.shape[0]
		self.topic_seed = np.random.random_integers(0,K-1,self.N)

		self.docid = [[i]*len(d) for i,d in enumerate(docs)]
		self.docid = np.array(list(itertools.chain(*self.docid)), dtype = np.int)

		self.alpha = 50/self.K
		self.beta = 200/self.V


	def set_priors(self,alpha,beta):

		"""
		Override default values for Dirichlet hyperparameters.

		alpha: hyperparameter for distribution of topics within documents.
		beta: hyperparameter for distribution of tokens within topics.
		"""

		assert type(alpha) == float and type(beta) == float

		self.alpha = alpha
		self.beta = beta


	def set_seed(self,seed):

		"""
		Override default values for random initial topic assignment, set to "seed" instead.
		"""

		assert seed.dtype==np.int and seed.shape==(self.N,)
		self.topic_seed = seed


	def set_sampled_topics(self,sampled_topics):

		"""
		Allocate sampled topics to the documents rather than estimate them.
		Automatically generate term-topic and document-topic matrices.
		"""

		assert sampled_topics.dtype == np.int and len(sampled_topics.shape) <= 2 	

		if len(sampled_topics.shape) == 1: self.sampled_topics = sampled_topics.reshape(1,sampled_topics.shape[0])
		else: self.sampled_topics = self.sampled_topics

		self.samples = self.sampled_topics.shape[0]

		self.tt = self.tt_comp(self.sampled_topics)
		self.dt = self.dt_comp(self.sampled_topics)


	def sample(self,burnin,thinning,samples,append=True):

		"""
		Estimate topics via Gibbs sampling.

		burnin: number of iterations to allow chain to burn in before sampling.
		thinning: thinning interval between samples.
		samples: number of samples to take.

		Total number of samples = burnin + thinning * samples

		If sampled topics already exist and append = True, extend chain from last sample.  
		If append = False, start new chain from the seed.
		"""

		if hasattr(self, 'sampled_topics') and append == True:

			sampled_topics = samplers.sampler(self.docid,self.tokens,self.sampled_topics[self.samples-1,:],
									self.N,self.V,self.K,self.D,self.alpha,self.beta,
									burnin,thinning,samples)

			tt_temp = self.tt_comp(sampled_topics)
			dt_temp = self.dt_comp(sampled_topics)

			self.sampled_topics = np.concatenate((self.sampled_topics,sampled_topics))
			self.tt = np.concatenate((self.tt,tt_temp),axis=2)
			self.dt = np.concatenate((self.dt,dt_temp),axis=2)

			self.samples = self.samples + samples

		else:

			self.samples = samples

			self.sampled_topics = samplers.sampler(self.docid,self.tokens,self.topic_seed,
										self.N,self.V,self.K,self.D,self.alpha,self.beta,
										burnin,thinning,self.samples)

			self.tt = self.tt_comp(self.sampled_topics)
			self.dt = self.dt_comp(self.sampled_topics)


	def dt_comp(self,sampled_topics):

		"""
		Compute document-topic matrix from sampled_topics.
		"""

		samples = sampled_topics.shape[0]
		dt = np.zeros((self.D,self.K,samples))

		for s in xrange(samples):
			dt[:,:,s] = samplers.dt_comp(self.docid, sampled_topics[s,:], self.N, self.K, self.D, self.alpha)

		return dt


	def tt_comp(self,sampled_topics):

		"""
		Compute term-topic matrix from sampled_topics.
		"""

		samples = sampled_topics.shape[0]
		tt = np.zeros((self.V,self.K,samples))

		for s in xrange(samples):
			tt[:,:,s] = samplers.tt_comp(self.tokens, sampled_topics[s,:], self.N, self.V, self.K, self.beta)

		return tt


	def topic_content(self,W):

		"""
		Print top W words in each topic to file.
		"""

		topic_top_words = []

		for t in xrange(self.K):
			top_word_indices = self.tt[:,t,self.samples-1].argsort()[-W:][::-1]
			topic_top_words.append([self.token_key.keys()[self.token_key.values().index(i)] for i in top_word_indices])

		with codecs.open("topic_description.csv","w","utf-8") as f:
			for t in xrange(self.K):
				words = ','.join(topic_top_words[t])
				f.write("topic" + str(t) + ',')
				f.write("%s\n" % words)


	def perplexity(self):

		"""
		Compute perplexity for each sample.
		"""

		return samplers.perplexity_comp(self.docid,self.tokens,self.tt,self.dt,self.N,self.K,self.samples)


	def samples_keep(self,index):

		"""
		Keep subset of samples.  If index is an integer, keep last N=index samples.  If index is a list, keep the samples
		corresponding to the index values in the list.
		"""

		if isinstance(index, (int, long)): index = range(self.samples)[-index:]

		self.sampled_topics = np.take(self.sampled_topics,index,axis=0)
		self.tt = np.take(self.tt,index,axis=2)
		self.dt = np.take(self.dt,index,axis=2)

		self.samples = len(index)


	def tt_avg(self,print_output=True):

		"""
		Compute average term-topic matrix, and print to file if print_output=True.
		"""		

		avg = self.tt.mean(axis=2)
		if print_output: np.savetxt("tt.csv", avg, delimiter = ",")
		return avg


	def dt_avg(self,print_output=True):

		"""
		Compute average document-topic matrix, and print to file if print_output=True.
		"""	

		avg = self.dt.mean(axis=2)
		if print_output: np.savetxt("dt.csv", avg, delimiter = ",")
		return avg


	def dict_print(self):

		"""
		Print mapping from tokens to numeric indices.
		"""	

		with codecs.open("dict.csv","w",encoding='utf-8') as f:
			for (v,k) in self.token_key.items(): f.write("%s,%d\n" % (v,k))


class Query():

	def __init__(self, docs, token_key, tt):

		"""
		Class for querying out-of-sample documents given output of estimated LDA model.

		docs: list (documents) of lists (string tokens within documents)
		token_key: mapping from tokens to numeric indices (tokens in docs not in key stripped out)
		tt: 3-d array of estimated topics (number of tokens in topic model x number of topics x number of samples)
		"""

		assert len(token_key) == tt.shape[0]

		ldatokens = set(token_key.keys())
		def keep(tokens): return [t for t in tokens if t in ldatokens]
		self.docs = map(keep,docs)

		self.D = len(self.docs)
		self.tt = tt
		self.V = tt.shape[0]
		self.K = tt.shape[1]
		self.samples = tt.shape[2]

		doc_list = list(itertools.chain(*self.docs))
		self.tokens = np.array([token_key[t] for t in doc_list], dtype = np.int)
		self.N = self.tokens.shape[0]
		self.topic_seed = np.random.random_integers(0, self.K-1, size = self.N)

		self.docid = [[i]*len(d) for i,d in enumerate(self.docs)]
		self.docid = np.array(list(itertools.chain(*self.docid)), dtype = np.int)

		self.alpha = 50/self.K


	def set_priors(self,alpha):

		"""
		Override default value for Dirichlet hyperparameter.

		alpha: hyperparameter for distribution of topics within documents.
		"""

		assert type(alpha) == float
		self.alpha = alpha


	def set_seed(self,seed):

		"""
		Override default values for random initial topic assignment, set to "seed" instead.
		seed is 2-d array (number of samples in LDA model x number of tokens in LDA model)
		"""

		assert seed.dtype==np.int and seed.shape==(self.samples,self.N)
		self.topic_seed = seed


	def query(self,query_samples):

		"""
		Query docs with query_samples number of Gibbs sampling iterations.
		"""

		self.sampled_topics = np.zeros((self.samples,self.N), dtype = np.int)

		for s in xrange(self.samples):

			self.sampled_topics[s,:] = samplers.sampler_query(self.docid, self.tokens, self.topic_seed,
											np.ascontiguousarray(self.tt[:,:,s], dtype=np.float),
											self.N, self.K, self.D, self.alpha, query_samples)

			print("Sample %d queried" % s)

		self.dt = np.zeros((self.D,self.K,self.samples))

		for s in xrange(self.samples):
			self.dt[:,:,s] = samplers.dt_comp(self.docid,self.sampled_topics[s,:], self.N, self.K, self.D, self.alpha)


	def perplexity(self):

		"""
		Compute perplexity for each sample.
		"""

		return samplers.perplexity_comp(self.docid,self.tokens,self.tt,self.dt,self.N,self.K,self.samples)


	def dt_avg(self,print_output=True):

		"""
		Compute average document-topic matrix, and print to file if print_output=True.
		"""	
		
		avg = self.dt.mean(axis=2)
		if print_output: np.savetxt("dt_query.csv", avg, delimiter = ",")
		return avg