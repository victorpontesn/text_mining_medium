# By: Victor Pontes (victoraleff@gmail.com)

import time 
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from .Text_Mining import TopicClustering
from scipy.sparse.linalg import svds
import faiss
import os
from .utils.preprocess import TextCleaner

def preprocess(data_path, process_columns, output_name_sufix, convert_parquet=True, remove_raw=False, clean_html=True):
    '''
    Processes the text of a files.

    Parameters
    ----------
        data_path (str): path of file in data directory.
        process_columns (list): list of strings with the name of the text columns to be processed.
        output_name (str): suffix for the output file name.
        remove_politcs (bool): apply remove_politics cleaner if True.
        remove_politcs (bool): apply remove_politics_covid cleaner if True.
    '''

    start  = time.time()

    file_name, file_type = data_path.split('.')
    assert file_type == 'csv', 'ERROR: pass a csv file'
    data = pd.read_csv(data_path)
    
    if convert_parquet:
        data.to_parquet(file_name+'.parquet', index=False)
        os.remove(data_path)

    Cleaner = TextCleaner() #create a cleaner object for compile regex paterns
    data.loc[:, process_columns] = data[process_columns].fillna('')\
                                                        .applymap(lambda x: Cleaner.transform(x, punctuation=True, accents=True, numbers=True,
                                                                                                 stopwords=True, html=clean_html, lower=True))
    data.to_parquet(f'{file_name}_{output_name_sufix}.parquet')

    if remove_raw:
        if os.path.exists(data_path): os.remove(data_path)
        if os.path.exists(file_name+'.parquet'): os.remove(file_name+'.parquet')
    print("Preprocessing Elapsed time: {:.2f} seconds".format(time.time() - start)) 

def bag_of_words(data_path, columns, name_bow_model = 'unigram',  save_tf=True, save_tfidf=False, 
                 ngram_range=(1,1), max_features=100000, min_df=15, max_df=0.5):
    '''
    Build term count matrices by bag of words (TF e TFIDF)

    Parameters
    ----------   
    
        data_path (str): path of parquet file with text data. 
        columns (str): list of strings with the name of the text columns that will be concatenated to generate the documents.
        name_bow_model (str): BoW model name that will name the output file.
        save_tf (bool) = if True, build and saves the term frequency matrix. 
        save_tfdf (bool) = if True, build and saves the tfidf matrix.

    '''

    start  = time.time()
    if not os.path.exists('models/'):
        os.makedirs('models')
    
    data = pd.read_parquet(data_path) 
    data['text'] = data[columns].apply(lambda x: ' '.join(x), axis=1)

    tf_vectorizer  = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, max_features=max_features, lowercase=False)
    tf_matrix    = tf_vectorizer.fit_transform(data.text)
    if save_tf: save_npz(f'models/tf_{name_bow_model}.npz',  tf_matrix)

    if save_tfidf:
        tfidf_transform = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
        tfidf_matrix = tfidf_transform.fit_transform(tf_matrix)
        save_npz(f'models/tfidf_{name_bow_model}.npz', tfidf_matrix)
    
    vocabulary = pd.Series(tf_vectorizer.get_feature_names())
    vocabulary.rename('ngrams').to_csv(f"models/vocabulary_{name_bow_model}.csv", index=False)

    print("Builder TFIDF Elapsed time: {:.2f} seconds".format(time.time() - start))

def builder_wals(bow_matrix_path='models/tfidf_unigram.npz', topic_model_name='unigram'):

    start  = time.time()
    if not os.path.exists('models/'):
        os.makedirs('models')

    bow_matrix = load_npz(bow_matrix_path)
    doc_topics, word_topics = TopicClustering.nnmf_wals(bow_matrix, n_components=300, weight=50.0)
    np.save(f'models/doc_topics_wals_{topic_model_name}.npy', doc_topics)
    np.save(f'models/word_topics_wals_{topic_model_name}.npy', word_topics)
    print("Builder WALS Topic Modeling Elapsed time: {:.2f} seconds".format(time.time() - start))

def builder_lsi(bow_matrix_path='tfidf_unigram.npz',  topic_model_name='unigram'):
    ''' 
        Realiza modelagem de tópicos com LSI
    '''
    start  = time.time()

    bow_matrix = load_npz(f'models/{bow_matrix_path}')
    
    U,  s,  Vt  = svds(bow_matrix, k=300)
    doc_topic   = bow_matrix @ Vt.T
    np.save(f'models/doc_topics_lsi_{topic_model_name}.npy', doc_topic)
    
    print("Builder LSI Elapsed time: {:.2f} seconds".format(time.time() - start))

def build_semantic_clustering(ncentroids=200, doc_topics_path='models/doc_topics_wals_unigram.npy', model_name='wals_unigram'):
    """
        Monta modelo de clusters semânticos utilizando o framework do Victor Pontes.
    """
    doc_top = np.load(doc_topics_path).astype('float32')
    faiss.normalize_L2(doc_top) 

    d = doc_top.shape[1]
    max_points_per_centroid = int(doc_top.shape[0]/ncentroids)+1
    kmeans_spherical = faiss.Kmeans(d, ncentroids, niter=25, nredo=3, verbose=True,  spherical=True, 
                                    min_points_per_centroid=20, max_points_per_centroid=max_points_per_centroid)
    kmeans_spherical.train(doc_top)
    centroids = kmeans_spherical.centroids

    dist_centroid_cosine, cluster_label_cosine = kmeans_spherical.index.search(doc_top, 1)

    np.savez(f"models/clusters_{model_name}_{ncentroids}_outputs.npz",
             distances=dist_centroid_cosine, 
             index=cluster_label_cosine,
             centroids=centroids,
             )

