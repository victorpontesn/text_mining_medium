from src.builder import preprocess, bag_of_words, builder_wals, build_semantic_clustering 

preprocess(data_path='data/articles.csv', process_columns = ['text', 'title'], 
           output_name_sufix='processed', clean_html=False)

bag_of_words(data_path='data/articles_processed.parquet', columns = ['text', 'title'], 
             name_bow_model='unigram_articles', save_tf=True, save_tfidf=True)

builder_wals(bow_matrix_path='models/tfidf_unigram_articles.npz', 
             topic_model_name='unigram_articles')

# Extensive Clustering
build_semantic_clustering(ncentroids=50, doc_topics_path='models/doc_topics_wals_unigram_articles.npy', 
                          model_name='extensive')

# Intensive Clustering
build_semantic_clustering(ncentroids=200, doc_topics_path='models/doc_topics_wals_unigram_articles.npy', 
                          model_name='intensive')