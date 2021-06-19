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
    def __init__(self, batch_size, max_len):
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

        def filter_max_length(x, y, max_length=max_len):
            """
            filter method
            """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = PT, EN

        # TRAIN DATA
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        self.data_train = self.data_train.cache()

        shu = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shu)
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

        # VALID DATA
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        pt: tf.Tensor containing the Portuguese sentence
        en: tf.Tensor containing the corresponding English
          sentence
        The tokenized sentences should include the start and end
          of sentence tokens
          The start token should be indexed as vocab_size
        The end token should be indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
        pt_tokens: tf.Tensor containing the Portuguese tokens
        en_tokens: tf.Tensor containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method
        """
        pt_tensor, en_tensor = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        pt_tensor.set_shape([None])
        en_tensor.set_shape([None])

        return pt_tensor, en_tensor