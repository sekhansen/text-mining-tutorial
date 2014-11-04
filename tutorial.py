"""
(c) 2014, Stephen Hansen, stephen.hansen@upf.edu

Python script for tutorial illustrating collapsed Gibbs sampling for Latent Dirichlet Allocation.

See explanation for commands on http://nbviewer.ipython.org/url/www.econ.upf.edu/~shansen/tutorial_notebook.ipynb.
"""

import itertools
import pandas as pd

import topicmodels

########## select data on which to run topic model #########

data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
data = data[data.year >= 1947]

########## clean documents #########

docsobj = topicmodels.RawDocs(data.speech, "stopwords.txt")
docsobj.token_clean(1)
docsobj.stopword_remove("tokens")
docsobj.stem()
docsobj.tf_idf("stems")
docsobj.stopword_remove("stems",5000)

print("number of stems = %d" % len(set(itertools.chain(*docsobj.stems))))
print("number of total words = %d" % len(list(itertools.chain(*docsobj.stems))))

########## estimate topic model #########

ldaobj = topicmodels.LDA(docsobj.stems,30)

ldaobj.sample(0,50,10)
ldaobj.sample(0,50,10)

ldaobj.samples_keep(4)
ldaobj.topic_content(20)

dt = ldaobj.dt_avg()
tt = ldaobj.dt_avg()
ldaobj.dict_print()

data = data.drop('speech',1)
for i in xrange(ldaobj.K): data['T' + str(i)] = dt[:,i]
data.to_csv("final_output.csv",index=False)

########## query aggregate documents #########

data['speech'] = [' '.join(s) for s in docsobj.stems]
aggspeeches = data.groupby(['year','president'])['speech'].apply(lambda x: ' '.join(x))
aggdocs = topicmodels.RawDocs(aggspeeches)

queryobj = topicmodels.Query(aggdocs.tokens,ldaobj.token_key,ldaobj.tt)
queryobj.query(10)
queryobj.perplexity()
queryobj.query(30)
queryobj.perplexity()

dt_query = queryobj.dt_avg()
aggdata = pd.DataFrame(dt_query,index=aggspeeches.index,columns=['T' + str(i) for i in xrange(queryobj.K)])
aggdata.to_csv("final_output_agg.csv")

########## top topics #########

def top_topics(x):
	top = x.values.argsort()[-5:][::-1]
	return(pd.Series(top,index=range(1,6)))

temp = aggdata.reset_index()
ranking = temp.set_index('president')
ranking = ranking - ranking.mean()
ranking = ranking.groupby(level='president').mean()
ranking = ranking.sort('year')
ranking = ranking.drop('year',1)
ranking = ranking.apply(top_topics,axis=1)
ranking.to_csv("president_top_topics.csv")