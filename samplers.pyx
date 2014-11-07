#cython: boundscheck=False
#cython: wraparound=False

"""
(c) 2014, Stephen Hansen, stephen.hansen@upf.edu

Cython script for collapsed Gibbs sampling for Latent Dirichlet Allocation (and related functions)
"""

from __future__ import division

from libc.stdlib cimport malloc, free

import time
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.int
FTYPE = np.float
ctypedef np.int_t DTYPE_t
ctypedef np.float_t FTYPE_t

#Using cython_gsl, we can directly call any gsl routine
from cython_gsl cimport *

cdef extern from "gsl/gsl_rng.h":
	ctypedef struct gsl_rng_type:
		pass
	ctypedef struct gsl_rng:
		pass
	gsl_rng_type *gsl_rng_mt19937
	gsl_rng *gsl_rng_alloc(gsl_rng_type * T)

	#Initialize seed 
	void gsl_rng_set(gsl_rng* r, unsigned long int seed)
	double gsl_rng_uniform(gsl_rng * r)
	

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

#Execute gsl_rng_set with a large number (time) which reasonably
#can be assumed to be unique to avoid similar sequences
gsl_rng_set(r, <unsigned long> time.time()*256)


cdef int multinomial_sample( double* p, int K ):

	"""
	Sample from multinomial distribution with probabilities p and length K
	"""

	cdef int new_topic
	cdef double rnd = gsl_rng_uniform(r)*p[K-1]

	for new_topic in xrange(K):
		if p[new_topic] > rnd:break

	return new_topic
	

def sampler(DTYPE_t [:] docs, 
	        DTYPE_t [:] tokens, 
	        DTYPE_t [:] topics, 
	        int N, int V, int K, int D,
	        double alpha, double beta,
	        int burnin, int thinning, int samples):

	"""
	Perform Gibbs sampling to sample topics, beginning from the seed "topics".

	docs: document identifier
	tokens: token indices
	N, V, K, D, alpha, beta: topic model parameters
	burnin: number of samples to burn in
	thinning: thinning interval
	samples: number of samples to take.

	output is (samples x N) array of topic assignments for each of the N tokens in the data
	"""

	cdef DTYPE_t [:, ::1] tok_topic_mat = np.zeros((V, K), dtype=DTYPE)
	cdef DTYPE_t [:] tok_topic_agg = np.zeros(K, dtype=DTYPE)
	cdef DTYPE_t [:, ::1] doc_topic_mat = np.zeros((D, K), dtype=DTYPE)

	cdef DTYPE_t [:, ::1] sampled_topics = np.zeros((samples, N), dtype=DTYPE)

	cdef int token, doc, old_topic, new_topic
	cdef int i, j
	cdef int iter = 0
	cdef int sample = 0
	cdef int maxiter = burnin + thinning*samples
	cdef double* p = <double*> malloc(K*sizeof(double))
	if not p:
		raise MemoryError()

	#initialize topic counts

	for i in xrange(N):
		tok_topic_mat[tokens[i],topics[i]] += 1
		doc_topic_mat[docs[i],topics[i]] += 1

	tok_topic_agg = np.sum(tok_topic_mat,axis=0)

	while iter < maxiter:

		for i in xrange(N):

			token = tokens[i]
			doc = docs[i]

			old_topic = topics[i]

			tok_topic_mat[token,old_topic] -= 1
			tok_topic_agg[old_topic] -= 1
			doc_topic_mat[doc,old_topic] -= 1

			for j in xrange(K): p[j] = (tok_topic_mat[token,j] + beta) / (tok_topic_agg[j] + beta*V) * (doc_topic_mat[doc,j] + alpha)
			for j in xrange(1,K): p[j] += p[j-1]

			new_topic = multinomial_sample(p,K)

			tok_topic_mat[token,new_topic] += 1
			tok_topic_agg[new_topic] += 1
			doc_topic_mat[doc,new_topic] += 1

			topics[i] = new_topic

		iter += 1
		if iter - burnin > 0 and (iter - burnin) % thinning == 0: 
			sampled_topics[sample,::1] = topics
			sample += 1
		print("Iteration %d of (collapsed) Gibbs sampling" % iter)

	free(p)

	return np.asarray(sampled_topics)


def sampler_query(DTYPE_t [:] docs, 
	        DTYPE_t [:] tokens,
	        DTYPE_t [:] topics,
	        FTYPE_t [:, ::1] tt,
	        int N, int K, int D,
	        double alpha, int samples): 

	"""
	Query documents represented by (docs,tokens,topics) using estimated term-topic matrix tt.

	After "samples" samples, return updated topic assignments.
	"""

	cdef DTYPE_t [:, ::1] doc_topic_mat = np.zeros((D, K), dtype=DTYPE)

	cdef int token, doc, old_topic, new_topic
	cdef int i, j
	cdef int sample = 0
	cdef double* p = <double*> malloc(K*sizeof(double))

	#initialize topic counts

	for i in xrange(N):
		doc_topic_mat[docs[i],topics[i]] += 1

	while sample < samples:

		for i in xrange(N):

			token = tokens[i]
			doc = docs[i]

			old_topic = topics[i]

			doc_topic_mat[doc,old_topic] -= 1

			for j in xrange(K): p[j] = tt[token,j] * (doc_topic_mat[doc,j] + alpha)
			for j in xrange(1,K): p[j] += p[j-1]

			new_topic = multinomial_sample(p,K)

			doc_topic_mat[doc,new_topic] += 1

			topics[i] = new_topic

		sample += 1

	free(p)

	return np.asarray(topics)


def tt_comp(DTYPE_t [:] tokens,
	        DTYPE_t [:] topics, 
	        int N, int V, int K,
	        double beta):

	"""
	Compute term-topic matrix from topic assignments
	"""

	cdef DTYPE_t [:, ::1] tok_topic_mat = np.zeros((V, K), dtype=DTYPE)
	cdef DTYPE_t [:] tok_topic_agg = np.zeros(K, dtype=DTYPE)
	cdef FTYPE_t [:, ::1] tt = np.zeros((V, K), dtype=FTYPE)
	cdef int i, v, k

	for i in xrange(N):
		tok_topic_mat[tokens[i],topics[i]] += 1

	tok_topic_agg = np.sum(tok_topic_mat,axis=0)

	for v in xrange(V):
		for k in xrange(K):
			tt[v,k] = (tok_topic_mat[v,k] + beta) / (tok_topic_agg[k] + V*beta) 

	return np.asarray(tt)


def dt_comp(DTYPE_t [:] docs,
	        DTYPE_t [:] topics, 
	        int N, int K, int D,
	        double alpha):

	"""
	Compute document-topic matrix from topic assignments
	"""

	cdef DTYPE_t [:, ::1] doc_topic_mat = np.zeros((D, K), dtype=DTYPE)
	cdef DTYPE_t [:] doc_topic_agg = np.zeros(D, dtype=DTYPE)
	cdef FTYPE_t [:, ::1] dt = np.zeros((D, K), dtype=FTYPE)
	cdef int i, d, k

	for i in xrange(N):
		doc_topic_mat[docs[i],topics[i]] += 1

	doc_topic_agg = np.sum(doc_topic_mat,axis=1)

	for d in xrange(D):
		for k in xrange(K):
			dt[d,k] = (doc_topic_mat[d,k] + alpha) / (doc_topic_agg[d] + K*alpha) 

	return np.asarray(dt)


def perplexity_comp(DTYPE_t [:] docs, 
	        DTYPE_t [:] tokens, 
	        FTYPE_t [:, :, ::1] tt,
	        FTYPE_t [:, :, ::1] dt,
	        int N, int K, int samples):

	"""
	Compute perplexity for each sample in "samples" given term-topic matrices "tt" and document-topic matrices "dt".
	"""

	cdef FTYPE_t [:] px 
	cdef FTYPE_t [:] perplexity = np.zeros(samples, dtype=FTYPE)
	cdef int doc_index,token_index
	cdef int n,k,s

	for s in xrange(samples):

		px = np.zeros(N, dtype=FTYPE)

		for n in xrange(N):

			doc_index = docs[n]
			token_index = tokens[n] 

			for k in xrange(K):
				px[n] = px[n] + tt[token_index,k,s]*dt[doc_index,k,s]

		px = np.log(px)

		perplexity[s] = np.exp( - np.sum(px) / N )

	return np.asarray(perplexity)