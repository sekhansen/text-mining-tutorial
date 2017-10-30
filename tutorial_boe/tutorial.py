"""
(c) 2015, Stephen Hansen, stephen.hansen@upf.edu

Python script for tutorial illustrating collapsed Gibbs sampling
for Latent Dirichlet Allocation.
"""

import pandas as pd
import numpy as np
import topicmodels

###############
# select data on which to run topic model
###############

data = pd.read_table("mpc_minutes.txt", encoding="utf-8")

###############
# bag of words
###############

data_agg = data.groupby('year').agg(lambda x: ' '.join(x))

docsobj = topicmodels.RawDocs(data_agg.minutes, "long")
docsobj.token_clean(1)
docsobj.stopword_remove("tokens")
docsobj.stem()
docsobj.stopword_remove("stems")
docsobj.term_rank("stems")

all_stems = [s for d in docsobj.stems for s in d]
print("number of unique stems = %d" % len(set(all_stems)))
print("number of total stems = %d" % len(all_stems))

bowobj = topicmodels.BOW(docsobj.stems)

data_agg['pos'] = bowobj.pos_count('stems')
data_agg['neg'] = bowobj.neg_count('stems')
data_agg['index'] = (data_agg.pos - data_agg.neg) /\
                    (data_agg.pos + data_agg.neg)

ons = pd.read_csv('ons_quarterly_gdp.csv')
data_agg['gdp_growth'] = ons.gdp_growth.values
data_agg['quarter'] = ons.quarter.values

temp = data_agg.groupby('quarter').agg(np.mean)
print(temp.corr())
temp['quarter'] = sorted(set(ons.label))
temp[['quarter', 'index', 'gdp_growth']].to_csv('index.csv', index=False)

###############
# clean documents for LDA
###############

docsobj = topicmodels.RawDocs(data.minutes, "long")
docsobj.token_clean(1)
docsobj.stopword_remove("tokens")
docsobj.stem()
docsobj.stopword_remove("stems")
docsobj.term_rank("stems")

all_stems = [s for d in docsobj.stems for s in d]
print("number of unique stems = %d" % len(set(all_stems)))
print("number of total stems = %d" % len(all_stems))

###############
# estimate topic model
###############

ldaobj = topicmodels.LDA.LDAGibbs(docsobj.stems, 20)
ldaobj.sample(2000, 50, 20)

# np.save('temp',ldaobj)
# ldaobj = np.matrix(np.load('temp.npy')).item(0)

ldaobj.topic_content(20, output_file="topic_description.csv")

###############
# query aggregate documents
###############

data['temp'] = [' '.join(s) for s in docsobj.stems]
aggspeeches = data.groupby(['year'])['temp'].apply(lambda x: ' '.join(x))
aggdocs = topicmodels.RawDocs(aggspeeches)

queryobj = topicmodels.LDA.QueryGibbs(aggdocs.tokens,
                                       ldaobj.token_key,
                                       ldaobj.tt)
queryobj.query(10)

dt_query = queryobj.dt_avg()
aggdata = pd.DataFrame(dt_query, index=aggspeeches.index,
                       columns=['T' + str(i) for i in range(queryobj.K)])
aggdata.to_csv("final_output_agg.csv")
