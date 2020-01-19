from tqdm import tqdm
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from collections import Counter
from scipy import spatial
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.special import gammaln
from collections import Counter

import imp, multiprocessing
import datetime
import LDA_ETM as lda
import scipy
import operator
import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import utils as my_utils
from sklearn.metrics import silhouette_score, davies_bouldin_score

grid = ['amazon_home_20000',
       'amazon_kindle_20000',
       'amazon_movies_20000']

def process_sampler(dataset_name):
    embedding_name = "glove_0.6"
    
    print(dataset_name, "entered")
    
    dataset = pd.read_pickle("datasets/"+ dataset_name + "_dataset")
    
    docs_edges = pickle.load(open("resources/"+ dataset_name + "_" +embedding_name + ".pickle","rb"))

    min_df = 5
    max_df = .5
    maxIters = 20

    beta=0.1
    alpha=0.1
    n_topics = 25
    lambda_param = 1.0
    maxiter = 20

    sampler = lda.LdaSampler(n_topics=n_topics, min_df=min_df, max_df=max_df, lambda_param=lambda_param,
                         alpha=alpha, beta=beta)

    sampler._initialize_(reviews = dataset.text.tolist())

    try:
        sampler.run(name=dataset_name, edge_dict=docs_edges, maxiter=maxiter, debug=False)
        joblib.dump(sampler, "dumps/mrf_lda/" + dataset_name + "_" + embedding_name + "_25topics")
        print(dataset_name, "dumped")
    except:
        print(dataset_name, "failed")
    
pool = multiprocessing.Pool(40)
pool.map(process_sampler, grid)
pool.close()