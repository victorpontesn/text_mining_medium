"""
    Module with specialized instructions for matrix factorization, topic modeling and clustering activities.
    By: Victor Pontes (victoraleff@gmail.com)
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
import faiss
from sklearn.metrics.pairwise import cosine_similarity

def sparse_info(matrix):
    """
    Prints some of the main features of a scipy sparse matrix.

    Parameters
    ----------
    matrix: scipy sparse array
    """
    nbits = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    mb = nbits/1024**2
    print('Shape: {} | Nonzero: {} | Memory Usage: {:.2f} Mb'.format(matrix.shape, matrix.nnz, mb))

def non_negative_als(X, n_components=300, alpha=1.0, n_iter=10, n_jobs=1):
    """
    performs non-negative matrix factorization with the Alternative 
    Least Square (ALS) method through successive interactions of the 
    algorithm provided by github.com/benfred/implicit library with 
    interspersed by inputting zeros to the negative values of both 
    factors in each interaction;

    Parameters
    ----------
    X: matrix to be factored.
    n_components (int, optional): components of latent space. Defaults to 300.
    alpha (float, optional): [description]. Defaults to 1.0.
    n_iter (int, optional): [description]. Defaults to 10.
    n_jobs (int, optional): [description]. Defaults to 1.

    Returns:
    ----------
    als.item_factors: left factor (document_topics for document_word input matrix)
    als.user_factors: right factor (word_topics for document_word input matrix)
    """
    als = AlternatingLeastSquares(factors=n_components, regularization=alpha,
                                   iterations=1, num_threads=n_jobs)
    for _ in tqdm(range(n_iter)):
        als.fit(X, show_progress=False)
        als.item_factors[als.item_factors < 0.0] = 0.0
        als.user_factors[als.user_factors < 0.0] = 0.0
    return (als.item_factors, als.user_factors) 
    
def nnmf_wals(X, n_components=300, alpha=1.0, n_jobs=4, n_iter=10, weight=50.0):
    """
    performs non-negative matrix factorization with the Weighted Alternative
    Least Square (WALS)

    Parameters
    ----------
    X: input matrix to be factored.
    n_components (int, optional): components of latent space. Defaults to 300.
    alpha (float, optional): [description]. Defaults to 1.0.
    n_iter (int, optional): [description]. Defaults to 10.
    n_jobs (int, optional): [description]. Defaults to 1.
    weight (float, optional): Defalts to 1.

    Returns
    ----------
    left factor: document_topics for document_word input matrix
    right factor: word_topics for document_word input matrix
    """
    wals_X = X.copy()
    wals_X.data = 1.0 + weight * wals_X.data
    left_factor, right_factor  = non_negative_als(wals_X, n_components=n_components, n_jobs=n_jobs, n_iter=n_iter)
    return left_factor, right_factor

def relevance_transform(pw_topic, lamb):
    """
    Transform the weigth of words in componentes from matrix words-topics in relevance score.
    The relevance score was proposed by Sievert e Shirley in 2014 in this paper:
    https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

    Obs: can be used for any vector representation of words (linear map of word space to any space) 
             example word-topic matrix or word-clusters matrix.
    
    Parameters
    ----------
    pw_topic: the word-topics matrix
    lamb: the coeficient

    Returns:
    word_topic_relevance: the transformed word-topics
    """
    pw_corpus = pw_topic.sum(axis=1) + 0.0000001
    pw_topic_div_corpus = np.divide(pw_topic.T, pw_corpus).T
    word_topic_relevance = lamb * np.log10(pw_topic) + (1-lamb) * np.log10(pw_topic_div_corpus)
    return word_topic_relevance

def view_topics(word_topic, word_names, n_top_words=10, n_top_top_words=5, relevance=True, lamb=0.5):
    """
    Viewing topics through your most important words.

        Obs: can be used for any vector representation of words (linear map of word space to any space) 
             example word-topic matrix or word-clusters matrix.

    Parameters
    ----------
        word_topic (numpy array or dense matrix): embbedings of words from topic model.
        word_names (list of strings): name of terms following indexing of matrix word_topic
        n_top_words (int, optional): number of top words presented in dataframe. Default to 10.
        n_top_top_words (int, optional): number of top words presented resumed list. Defaults to 5.
                                         n_top_top_words < n_top_words
        relevance (boolean, optional): define the method of ranking most important words,if True use 
                                       relevance score, if False use the weigth of words in componentes

    Returns:
    ----------
        topwords_df: dataframe with n_top_words per topic
        right factor: pandas Series with n_top_top_words per tipic 
    """
    np.seterr(divide='ignore')
    if relevance: word_topic = relevance_transform(word_topic, lamb) 
    array_sorted = np.argsort(word_topic, axis=0)
    topwords_inverse = array_sorted[-n_top_words:]
    topwords = np.flipud(topwords_inverse)
    topwords_df = pd.DataFrame(topwords)
    topwords_df = topwords_df.applymap(lambda x: word_names[x])
    top_top_words = topwords_df.T.iloc[:,:n_top_top_words].fillna('').apply(lambda x: ' | '.join(x), axis=1)
    return topwords_df, top_top_words

def get_word_cluster_matrix(cluster_labels, tf_doc_word):
    """
    Creates the word-cluster matrix whose values represent the frequency of words in all documents 
    associated with each cluster.

    Parameters
    ----------
    tf_doc_word (scipy sparse matrix): term-frequency matrix
    cluster_labels (iterator): clusters labels associated with each document following rows indexing of tf_doc_word

    Returns
    -------
    word_cluster_tf (numpy array): word_cluster term-frequency matrix
    """

    DF = pd.DataFrame({'cluster_label': cluster_labels, 'idx': np.arange(cluster_labels.shape[0])})
    cluster_docs_idx = DF.groupby('cluster_label')['idx'].apply(list)
    cluster_term_frequency = cluster_docs_idx.apply(lambda x: tf_doc_word[x].sum(axis=0).tolist()[0])
    word_cluster_tf = np.array(cluster_term_frequency.tolist()).T
    return word_cluster_tf    


class FaissNearestNeighbors:
    """
    Unsupervised learner for implementing neighbor searches.
    This algorithm was implemented from the faiss library, following the sklearn.neighbors.NearestNeighbors scheme, as a faster alternative.
    """
    def __init__(self):
        "Initialize self."
        self.index = None

    def fit(self, X):
        """Fit the model using X as training data
        
        Parameters
        ----------
        X (array-like, sparse matrix): training data
        """
        d = X.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X=None, n_neighbors=5, return_distance=True):
        """
        Finds the K-neighbors of a point according to Euclidean distance. 
        Returns indices of and distances to the neighbors of each point.
        
        Parameters
        ----------
        X (array-like, sparse matrix): the query points or points.
        n_neighbors (int): number of neighbors to get. Defaults to 5.
        return_distence (boolean): if False, distance will not be returned. Defaults to True.

        Returns
        -------
        distances (array): euclidean distances of the nearest points in training matrix
        indices (array): indices of the nearest points in training matrix.
        """
        distances, indices = self.index.search(X.astype(np.float32), n_neighbors)
        return distances, indices if return_distance else indices