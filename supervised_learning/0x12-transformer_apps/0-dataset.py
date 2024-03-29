#!/usr/bin/env python3
"""
Dataset Class
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and Preps a model for machine translation
    """
    def __init__(self):
        """
        data_train: contains the ted_hrlr_translate/pt_to_en
          tf.data.Dataset train split, loaded as_supervided
        data_valid: contains the ted_hrlr_translate/pt_to_en
          tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt: Portuguese tokenizer created from the
          training set
        tokenizer_en: English tokenizer created from the
          training set
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = PT, EN

    def tokenize_dataset(self, data):
        """
        data: tf.data.Dataset whose examples are formatted
          as a tuple (pt, en)
        pt: tf.Tensor containing the Portuguese sentence
        en: tf.Tensor containing the corresponding English
            sentence
        The maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
        tokenizer_pt: Portuguese tokenizer
        tokenizer_en: English tokenizer
        """
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2 ** 15)

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2 ** 15)

        return (tokenizer_pt, tokenizer_en)
