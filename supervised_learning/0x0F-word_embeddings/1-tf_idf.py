#!/usr/bin/env python3
"""
TF-IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding
    sentences : list of sentences to analyze
    vocab : list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used
    Returns: embeddings, features
    embeddings : numpy.ndarray of shape (s, f) containing the embeddings
    s: number of sentences in sentences
    f: number of features analyzed
    features : list of the features used for embeddings
    """
    tf_idf = TfidfVectorizer(vocabulary=vocab)
    X = tf_idf.fit_transform(sentences)
    features = tf_idf.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
