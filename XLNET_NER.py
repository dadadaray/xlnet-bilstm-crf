#! usr/bin/env python3
# -*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import io
import pickle

import re
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from absl import flags, logging
from xlnet import modeling
from xlnet import xlnet
# from xlnet import optimization
# from xlnet import tokenization
from xlnet import prepro_utils
import tensorflow as tf
import metrics
from tensorflow.contrib.layers.python.layers import initializers
from xlnet import model_utils
import numpy as np
from lstm_crf_layer import BLSTM_CRF

FLAGS = flags.FLAGS
import time
import json
import sentencepiece as sp
from tensorflow.contrib.layers.python.layers import initializers


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
MIN_FLOAT = -1e30
start=time.time()
## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "xlnet_config_file", None,
    "The config json file corresponding to the pre-trained xlnet model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "export_dir", None,
    "export_dir")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "prefile", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "save_steps", None,
    "save_steps")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "model_dir", None,
    "model_dir")

flags.DEFINE_bool(
    "do_export", True,
    "do_export")

flags.DEFINE_bool(
    "use_bfloat16", False,
    "use_bfloat16")
# if you download cased checkpoint you should use "False",if uncased you should use
# "True"
# if we used in bio-medical field，don't do lower case would be better!

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_float(
    "dropout", 0.1,
    "dropout")

flags.DEFINE_float(
    "dropatt", 0.1,
    "dropatt")

flags.DEFINE_string(
    "init", "normal",
    "init")

flags.DEFINE_string(
    "decay_method", "poly",
    "decay_method")

flags.DEFINE_string(
    "predict_tag", "这个很好",
    "predict_tag")

flags.DEFINE_float(
    "init_range", 0.1,
    "init_range")

flags.DEFINE_float(
    "init_std", 0.02,
    "init_range")

flags.DEFINE_float(
    "min_lr_ratio", 0.0,
    "min_lr_ratio")

flags.DEFINE_float(
    "weight_decay", 0.15,
    "weight_decay")
flags.DEFINE_float(
    "adam_epsilon", 1e-8,
    "adam_epsilon")

flags.DEFINE_float(
    "clip", 1.0,
    "clip")

flags.DEFINE_integer(
    "clamp_len", -1,
    "clamp_len")

flags.DEFINE_integer(
    "warmup_steps", 0,
    "warmup_steps")

flags.DEFINE_integer(
    "train_steps", 100000,
    "warmup_steps")

flags.DEFINE_integer(
    "iterations", 1000,
    "iterations")

flags.DEFINE_integer(
    "num_hosts", 1,
    "num_hosts")

flags.DEFINE_integer(
    "max_save", 100000,
    "max_save")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_core_per_host", 1,
    "num_core_per_host")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "do_pre_eval", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")
flags.DEFINE_bool("crf", True, "use crf!")



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_masks,
                 segment_ids,
                 label_ids):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def remove(text):
        remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        return re.sub(remove_chars, ' ', text)

    @classmethod
    def _read_data(cls, input_file):
        """Read a BIO data!"""
        rf = io.open(input_file, 'r', encoding='utf-8')
        lines = [];
        words = [];
        labels = [];

        num_words=0
        num_sents=0
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if len(line.strip()) == 0: #一句话的结束
                num_sents+=1
                num_words+=len(words)
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l, w))
                words = []
                labels = []

            words.append(word)
            labels.append(label)
        rf.close()
        print("numbers of words",num_words)
        print("numbers of sentences", num_sents)
        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "eng.train.openNLP")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "eng.testa.openNLP")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "eng.testb.openNLP")), "test"
        )
    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return [ "<pad>", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O", "X",
                "<cls>", "<sep>"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = prepro_utils.convert_to_unicode(line[1])
            labels = prepro_utils.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


# XLNET 分词类

class XLNetTokenizer(object):
    """Default text tokenizer for XLNet"""

    def __init__(self,
                 sp_model_file,
                 lower_case=False):
        """Construct XLNet tokenizer"""
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(sp_model_file)
        self.lower_case = lower_case

    def tokenize(self, text):
        """Tokenize text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        tokenized_pieces = prepro_utils.encode_pieces(self.sp_processor, processed_text)
        return tokenized_pieces

    def encode(self, text):
        """Encode text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        encoded_ids = prepro_utils.encode_ids(self.sp_processor, processed_text)
        return encoded_ids

    def token_to_id(self,
                    token):
        """Convert token to id for XLNet"""
        return self.sp_processor.PieceToId(token)

    def id_to_token(self,
                    id):
        """Convert id to token for XLNet"""
        return self.sp_processor.IdToPiece(id)

    def tokens_to_ids(self,
                      tokens):
        """Convert tokens to ids for XLNet"""
        return [self.sp_processor.PieceToId(token) for token in tokens]

    def ids_to_tokens(self,
                      ids):
        """Convert ids to tokens for XLNet"""
        return [self.sp_processor.IdToPiece(id) for id in ids]


# XLNET 数据预处理与模型

class XLNetExampleConverter(object):
    """Default example converter for XLNet"""

    def __init__(self,
                 label_list,
                 max_seq_length,
                 tokenizer):
        """Construct XLNet example converter"""
        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i

        self.segment_vocab_list = ["<a>", "<b>", "<cls>", "<sep>", "<pad>"]
        self.segment_vocab_map = {}
        for (i, segment_vocab) in enumerate(self.segment_vocab_list):
            self.segment_vocab_map[segment_vocab] = i

        self.label_list = label_list
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def convert_single_example(self, example, logging=True):
        '''
        对单个样本进行分析, 然后将字转化为id，标签转化为id，然后结构化到InputFeature中
        :param example:
        :param logging:
        :return:
        '''
        # processors = {"ner": NerProcessor}
        # label_list = processor.get_labels()

        default_feature = InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_masks=[1] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_ids=[0] * self.max_seq_length)

        if isinstance(example, PaddingInputExample):
            return default_feature

        #token_items = self.tokenizer.tokenize(example.text)
        textlist = example.text.split(' ')
        label_items = example.label.split(' ')


        # if len(label_items) != len([token for token in token_items if token.startswith(prepro_utils.SPIECE_UNDERLINE)]):
        #     return default_feature

        tokens = []
        labels = []
        #idx = 0

        for i, (word, label) in enumerate(zip(textlist, label_items)):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i, _ in enumerate(token):
                if i == 0:
                    labels.append(label)
                else:
                    labels.append("X")

        # print("len of tokens2",len(tokens))
        # for token in textlist:
        #     tokn=self.tokenizer.tokenize(token)
        #     tokens.append(tokn)
        #     if token.startswith(prepro_utils.SPIECE_UNDERLINE):
        #         label = label_items[idx]
        #         idx += 1
        #     else:
        #         label = "X"
        #
        #
        #     labels.append(label)

        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[0:(self.max_seq_length - 2)]

        if len(labels) > self.max_seq_length - 2:
            labels = labels[0:(self.max_seq_length - 2)]

        # for (i,token) in enumerate(tokens):
        #     print("token:",tokens[i])
        #     print("label",labels[i])

        printable_tokens = [prepro_utils.printable_text(token) for token in tokens]

        # The convention in XLNet is:
        # (a) For sequence pairs:
        #  tokens:      is it a dog ? [SEP] no , it is not . [SEP] [CLS]
        #  segment_ids: 0  0  0 0   0 0     1  1 1  1  1   1 1     2
        # (b) For single sequences:
        #  tokens:      this dog is big . [SEP] [CLS]
        #  segment_ids: 0    0   0  0   0 0     2
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the last vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense when
        # the entire model is fine-tuned.
        input_tokens = []
        segment_ids = []
        label_ids = []

        for i, token in enumerate(tokens):
            input_tokens.append(token)
            segment_ids.append(self.segment_vocab_map["<a>"])
            label_ids.append(self.label_map[labels[i]])

        # input_tokens.append("<sep>")
        # segment_ids.append(self.segment_vocab_map["<a>"])
        # label_ids.append(self.label_map["<sep>"])

        input_tokens.append("<cls>")
        segment_ids.append(self.segment_vocab_map["<cls>"])
        label_ids.append(self.label_map["<cls>"])

        input_ids = self.tokenizer.tokens_to_ids(input_tokens)

        # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
        input_masks = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        # while len(input_ids) < self.max_seq_length:
        #     input_ids.append(0)
        #     input_masks.append(0)
        #     segment_ids.append(0)
        #     label_ids.append(0)

        if len(input_ids) < self.max_seq_length:
            pad_seq_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [self.special_vocab_map["<pad>"]] * pad_seq_length
            input_masks = input_masks + [1] * pad_seq_length
            segment_ids = segment_ids + [self.segment_vocab_map["<pad>"]] * pad_seq_length
            label_ids = label_ids + [self.label_map["<pad>"]] * pad_seq_length
            input_tokens = input_tokens + ["<pad>"] * pad_seq_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_masks) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        assert len(input_masks) == self.max_seq_length

        if logging:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("labels: %s" % " ".join(labels))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_masks: %s" % " ".join([str(x) for x in input_masks]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        feature = InputFeatures(
            input_ids=input_ids,
            input_masks=input_masks,
            segment_ids=segment_ids,
            label_ids=label_ids)

        return feature

    def convert_single_example_2return(self, example, logging=True):
        '''
        对单个样本进行分析, 然后将字转化为id，标签转化为id，然后结构化到InputFeature中
        :param example:
        :param logging:
        :return:
        '''
        # processors = {"ner": NerProcessor}
        # label_list = processor.get_labels()

        default_feature = InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_masks=[1] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_ids=[0] * self.max_seq_length)

        if isinstance(example, PaddingInputExample):
            return default_feature



        #token_items = self.tokenizer.tokenize(example.text)
        textlist = example.text.split(' ')
        label_items = example.label.split(' ')


        # if len(label_items) != len([token for token in token_items if token.startswith(prepro_utils.SPIECE_UNDERLINE)]):
        #     return default_feature

        tokens = []
        labels = []
        #idx = 0

        for i, (word, label) in enumerate(zip(textlist, label_items)):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i, _ in enumerate(token):
                if i == 0:
                    labels.append(label)
                else:
                    labels.append("X")


        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[0:(self.max_seq_length - 2)]

        if len(labels) > self.max_seq_length - 2:
            labels = labels[0:(self.max_seq_length - 2)]

        # for (i,token) in enumerate(tokens):
        #     print("token:",tokens[i])
        #     print("label",labels[i])

        printable_tokens = [prepro_utils.printable_text(token) for token in tokens]

        # The convention in XLNet is:
        # (a) For sequence pairs:
        #  tokens:      is it a dog ? [SEP] no , it is not . [SEP] [CLS]
        #  segment_ids: 0  0  0 0   0 0     1  1 1  1  1   1 1     2
        # (b) For single sequences:
        #  tokens:      this dog is big . [SEP] [CLS]
        #  segment_ids: 0    0   0  0   0 0     2
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the last vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense when
        # the entire model is fine-tuned.
        input_tokens = []
        segment_ids = []
        label_ids = []

        for i, token in enumerate(tokens):
            input_tokens.append(token)
            segment_ids.append(self.segment_vocab_map["<a>"])
            label_ids.append(self.label_map[labels[i]])

        # input_tokens.append("<sep>")
        # segment_ids.append(self.segment_vocab_map["<a>"])
        # label_ids.append(self.label_map["<sep>"])

        input_tokens.append("<cls>")
        segment_ids.append(self.segment_vocab_map["<cls>"])
        label_ids.append(self.label_map["<cls>"])

        input_ids = self.tokenizer.tokens_to_ids(input_tokens)

        # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
        input_masks = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        # while len(input_ids) < self.max_seq_length:
        #     input_ids.append(0)
        #     input_masks.append(0)
        #     segment_ids.append(0)
        #     label_ids.append(0)

        if len(input_ids) < self.max_seq_length:
            pad_seq_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [self.special_vocab_map["<pad>"]] * pad_seq_length
            input_masks = input_masks + [1] * pad_seq_length
            segment_ids = segment_ids + [self.segment_vocab_map["<pad>"]] * pad_seq_length
            label_ids = label_ids + [self.label_map["<pad>"]] * pad_seq_length
            input_tokens = input_tokens + ["<pad>"] * pad_seq_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_masks) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        assert len(input_masks) == self.max_seq_length

        if logging:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("labels: %s" % " ".join(labels))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_masks: %s" % " ".join([str(x) for x in input_masks]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        feature = InputFeatures(
            input_ids=input_ids,
            input_masks=input_masks,
            segment_ids=segment_ids,
            label_ids=label_ids)

        return feature, input_tokens, label_ids

    def convert_examples_to_features(self, examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in enumerate(examples):
            if idx % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

            feature = self.convert_single_example(example, logging=(idx < 5))
            features.append(feature)

        return features

    def convert_examples_to_features2return(self, examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        batch_tokens = []
        batch_labels = []
        for (idx, example) in enumerate(examples):
            if idx % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

            feature,ntokens, label_ids = self.convert_single_example_2return(example, logging=(idx < 5))
            features.append(feature)
            batch_tokens.extend(ntokens)
            batch_labels.extend(label_ids)
        print("len of grenerate dev tokens",len(batch_tokens))
        return features,batch_tokens, batch_labels

    def file_based_convert_examples_to_features(self, examples, output_file):
        '''
        将数据转化为TF_Record 结构，作为模型数据输入
        :param examples:
        :param output_file: tf_record数据
        :return:
        '''

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


        with tf.python_io.TFRecordWriter(output_file) as writer:
            alltokens = []
            for (idx, example) in enumerate(examples):
                if idx % 10000 == 0:
                    tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

                feature,token,lable = self.convert_single_example_2return(example, logging=(idx < 5))
                # alltokens.append(token)

                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_masks"] = create_float_feature(feature.input_masks)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label_ids"] = create_int_feature(feature.label_ids)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
            # print("aaaaaaaaaaaaaaaaThis number of all tokens",len(alltokens))


class XLNetInputBuilder(object):
    """Default input builder for XLNet"""

    @staticmethod
    def get_input_builder(features,
                          seq_length,
                          is_training,
                          drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        all_input_ids = []
        all_input_masks = []
        all_segment_ids = []
        all_label_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_masks.append(feature.input_masks)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)

        def input_fn(params,
                     input_context=None):
            batch_size = params["batch_size"]
            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "input_masks": tf.constant(all_input_masks, shape=[num_examples, seq_length], dtype=tf.float32),
                "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "label_ids": tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            })

            if input_context is not None:
                tf.logging.info("Input pipeline id %d out of %d", input_context.input_pipeline_id,
                                input_context.num_replicas_in_sync)
                d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))

            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            return d

        return input_fn

    @staticmethod
    def get_file_based_input_fn(input_file,
                                seq_length,
                                is_training,
                                drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_masks": tf.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }

        def _decode_record(record,
                           name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params, input_context=None):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)

            if input_context is not None:
                tf.logging.info("Input pipeline id %d out of %d", input_context.input_pipeline_id,
                                input_context.num_replicas_in_sync)
                d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))

            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

            return d

        return input_fn

    @staticmethod
    def get_serving_input_fn(seq_length):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        def serving_input_fn():
            with tf.variable_scope("serving"):
                features = {
                    'input_ids': tf.placeholder(tf.int32, [None, seq_length], name='input_ids'),
                    'input_masks': tf.placeholder(tf.float32, [None, seq_length], name='input_masks'),
                    'segment_ids': tf.placeholder(tf.int32, [None, seq_length], name='segment_ids')
                }

                return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()

        return serving_input_fn


# xlnet 模型建立
class XLNetModelBuilder(object):
    """Default model builder for XLNet"""

    def __init__(self,
                 model_config,
                 use_tpu=False):
        """Construct XLNet model builder"""
        self.model_config = model_config
        self.use_tpu = use_tpu

    def _get_masked_data(self,
                         data_ids,
                         label_list):
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        pad_id = tf.constant(label_map["<pad>"], shape=[], dtype=tf.int32)
        out_id = tf.constant(label_map["O"], shape=[], dtype=tf.int32)
        x_id = tf.constant(label_map["X"], shape=[], dtype=tf.int32)
        cls_id = tf.constant(label_map["<cls>"], shape=[], dtype=tf.int32)
        sep_id = tf.constant(label_map["<sep>"], shape=[], dtype=tf.int32)

        masked_data_ids = (tf.cast(tf.not_equal(data_ids, pad_id), dtype=tf.int32) *
                           tf.cast(tf.not_equal(data_ids, out_id), dtype=tf.int32) *
                           tf.cast(tf.not_equal(data_ids, x_id), dtype=tf.int32) *
                           tf.cast(tf.not_equal(data_ids, cls_id), dtype=tf.int32) *
                           tf.cast(tf.not_equal(data_ids, sep_id), dtype=tf.int32))

        return masked_data_ids

    def _create_model(self,
                      input_ids,
                      input_masks,
                      segment_ids,
                      label_ids,
                      label_list,
                      mode):
        # 写入label2id:
        label_map = {}
        # here start with zero this means that "[PAD]" is zero
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        with open(FLAGS.middle_output + "/label2id.pkl", 'wb') as w:
            pickle.dump(label_map, w)

        #print("shaple of inputid", input_ids.shape)
        """Creates XLNet-NER model"""
        model = xlnet.XLNetModel(
            xlnet_config=self.model_config,
            run_config=xlnet.create_run_config(mode == tf.estimator.ModeKeys.TRAIN, True, FLAGS),
            # input_ids=input_ids,
            # input_mask=input_masks,
            # seg_ids=segment_ids
            input_ids = tf.transpose(input_ids, perm=[1, 0]),
            input_mask = tf.transpose(input_masks, perm=[1, 0]),
            seg_ids = tf.transpose(segment_ids, perm=[1, 0]))

        # 不添加lstm
        # output_layer = model.get_sequence_output()
        # # output_layer shape is
        # if is_training:
        #     output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
        # logits = hidden2tag(output_layer, num_labels)
        # # TODO test shape
        # logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # if FLAGS.crf:
        #     mask2len = tf.reduce_sum(mask, axis=1)
        #     loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
        #     predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        #     return (loss, logits, predict)
        #
        # else:
        #     loss, predict = softmax_layer(logits, labels, num_labels, mask)
        #
        #     return (loss, logits, predict)


        #xlnet 不加BILSTM_CRF写法
        # initializer = model.get_initializer()
        #
        # with tf.variable_scope("ner", reuse=tf.AUTO_REUSE):
        #     result = tf.transpose(model.get_sequence_output(), perm=[1, 0, 2])
        #     result_mask = tf.cast(tf.expand_dims(1 - input_masks, axis=-1), dtype=tf.float32)
        #
        #     dense_layer = tf.keras.layers.Dense(units=len(label_list), activation=None, use_bias=True,
        #                                         kernel_initializer=initializer, bias_initializer=tf.zeros_initializer,
        #                                         kernel_regularizer=None, bias_regularizer=None, trainable=True)
        #
        #     dropout_layer = tf.keras.layers.Dropout(rate=0.1, seed=np.random.randint(10000))
        #
        #     result = dense_layer(result)
        #     if mode == tf.estimator.ModeKeys.TRAIN:
        #         result = dropout_layer(result)
        #
        #     masked_predict = result * result_mask + MIN_FLOAT * (1 - result_mask)
        #     predict_ids = tf.cast(tf.argmax(tf.nn.softmax(masked_predict, axis=-1), axis=-1), dtype=tf.int32)
        #
        # loss = tf.constant(0.0, dtype=tf.float32)
        # if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL] and label_ids is not None:
        #     with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        #         label = tf.cast(label_ids, dtype=tf.float32)
        #         label_mask = tf.cast(1 - input_masks, dtype=tf.float32)
        #         masked_label = tf.cast(label * label_mask, dtype=tf.int32)
        #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_label,
        #                                                                        logits=masked_predict)
        #         loss = tf.reduce_sum(cross_entropy * label_mask) / tf.reduce_sum(tf.reduce_max(label_mask, axis=-1))
        #
        # return loss, predict_ids

       #  #添加BILSTM_CRF
        embedding = model.get_sequence_output()
        embeddings = tf.transpose(embedding, perm=[1,0,2])

        #output_layer = tf.keras.layers.Dropout(rate=0.1)(embeddings)
        max_seq_length = embeddings.shape[1].value


        # 算序列真实长度
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        # 添加bilstm crf output layer
        lstm_size=128
        cell='lstm'
        num_layers=1
        dropout_rate=0.9

        blstm_crf = BLSTM_CRF(embedded_chars=embeddings, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                              dropout_rate=dropout_rate, initializers=initializers, num_labels=len(label_list),
                              seq_length=max_seq_length, labels=label_ids, lengths=lengths, is_training=True)
       #使用lstm+crf
        rst = blstm_crf.add_blstm_crf_layer(crf_only=False)
        return rst




    def get_model_fn(self, label_list):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features,
                     labels,
                     mode,
                     params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            def metric_fn(label_ids,
                          predict_ids):
                precision = tf.metrics.precision(labels=label_ids, predictions=predict_ids)
                recall = tf.metrics.recall(labels=label_ids, predictions=predict_ids)

                metric = {
                    "precision": precision,
                    "recall": recall,
                }

                return metric

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_masks = features["input_masks"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"] if mode in [tf.estimator.ModeKeys.TRAIN,
                                                          tf.estimator.ModeKeys.EVAL] else None

            loss, predict_ids = self._create_model(input_ids, input_masks, segment_ids, label_ids, label_list, mode)

            scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op, _, _ = model_utils.get_train_op(FLAGS, loss)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:
                masked_label_ids = self._get_masked_data(label_ids, label_list)
                masked_predict_ids = self._get_masked_data(predict_ids, label_list)
                eval_metrics = (metric_fn, [masked_label_ids, masked_predict_ids])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"predict": predict_ids},
                    scaffold_fn=scaffold_fn)

            return output_spec

        return model_fn


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    """
    # print("example",example)
    label_map = {}
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)

    default_feature = InputFeatures(
        input_ids=[0] * max_seq_length,
        input_masks=[1] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_ids=[0] * max_seq_length)

    if isinstance(example, PaddingInputExample):
        return default_feature

    # example.text=prepro_utils.convert_to_unicode(example.text)
    # print("this is exmaple",example.text)
    # print("this is type",type(example.text))
    textlist = tokenizer.tokenize(example.text)
    # print("myexample:",tokenizer.tokenize("this is a test to test if it's error!"))
    labellist = example.label.split(" ")
    # print("lablelist",labellist)
    # print(len(labellist))
    # textlist=example.text.split(' ')

    # textlist=example.text
    # print(" this is textlist",textlist)
    # print(len(textlist))

    if len(labellist) != len([token for token in textlist if token.startswith(prepro_utils.SPIECE_UNDERLINE)]):
        return default_feature

    # labellist = example.label.split(" ")
    tokens = []
    labels = []
    idx = 0

    # import chardet
    for token in textlist:

        # encode=chardet.detect(token.encode())
        # print("token",token)
        # print("encode1",encode)
        if token.startswith(prepro_utils.SPIECE_UNDERLINE):
            label = labellist[idx]
            idx += 1
        else:
            label = "X"

        tokens.append(token)
        labels.append(label)


    # only Account for [CLS] with "- 2".
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    if len(labels) > max_seq_length - 2:
        labels = labels[0:(max_seq_length - 2)]

    # printable_token=tokens
    printable_token = [prepro_utils.printable_text(token) for token in tokens]

    # The convention in XLNet is:
    # (a) For sequence pairs:
    #  tokens:      is it a dog ? [SEP] no , it is not . [SEP] [CLS]
    #  segment_ids: 0  0  0 0   0 0     1  1 1  1  1   1 1     2
    # (b) For single sequences:
    #  tokens:      this dog is big . [SEP] [CLS]
    #  segment_ids: 0    0   0  0   0 0     2
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the last vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense when
    # the entire model is fine-tuned.

    ntokens = []
    segment_ids = []
    label_ids = []

    # 添加map
    special_vocab_list = ["<pad>","<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<eod>", "<eop>"]
    special_vocab_map = {}
    for (i, special_vocab) in enumerate(special_vocab_list):
        special_vocab_map[special_vocab] = i

    segment_vocab_list = ["<pad>","<a>", "<b>", "<cls>", "<sep>"]
    segment_vocab_map = {}
    for (i, segment_vocab) in enumerate(segment_vocab_list):
        segment_vocab_map[segment_vocab] = i

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(segment_vocab_map["<a>"])
        label_ids.append(label_map[labels[i]])

    # ntokens.append("<sep>")
    # segment_ids.append(segment_vocab_map["<a>"])
    # label_ids.append(label_map["<sep>"])

    ntokens.append("<cls>")
    segment_ids.append(segment_vocab_map["<cls>"])
    label_ids.append(label_map["<cls>"])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.

    # segment_ids.append(segment_vocab_map["<a>"])
    # label_ids.append(label_map["<sep>"])

    # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are   attended to.
    input_ids = tokenizer.tokens_to_ids(ntokens)
    input_masks = [1] * len(input_ids)

    # use zero to padding and you should
    if len(input_ids) < max_seq_length:
        pad_seq_length = max_seq_length - len(input_ids)
        input_ids = [special_vocab_map["<pad>"]] * pad_seq_length + input_ids
        input_masks = [1] * pad_seq_length + input_masks
        segment_ids = [segment_vocab_map["<pad>"]] * pad_seq_length + segment_ids
        label_ids = [label_map["<pad>"]] * pad_seq_length + label_ids

    assert len(input_ids) == max_seq_length
    assert len(input_masks) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if logging:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(printable_token))
        tf.logging.info("labels: %s" % " ".join(labels))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_masks: %s" % " ".join([str(x) for x in input_masks]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_masks=input_masks,
        segment_ids=segment_ids,
        label_ids=label_ids)
    # we need ntokens because if we do predict it can help us return to original token.
    return feature, ntokens, label_ids


def convert_examples_to_features(examples, labellist, max_seq_length, tokenizer, mode=None):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (idx, example) in enumerate(examples):
        if idx % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

        feature, _, _ = XLNetExampleConverter.convert_single_example_2return(example)
        features.append(feature)

    return features


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    # writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    features = []
    xlnetExampleCon = XLNetExampleConverter(label_list, max_seq_length, tokenizer)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids = xlnetExampleCon.convert_single_example_2return(example)
        features.append(feature)
        # print("the len of example", len(examples))
        # print("the len of each ntokens", len(ntokens))
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)


    return features,batch_tokens, batch_labels



def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    # TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss, transition


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(xlnet_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=xlnet.create_run_config(is_training, True, FLAGS),
        # is_training=is_training,
        # input_ids=tf.transpose(input_ids, perm=[1,0]),  #int
        # seg_ids=tf.transpose(segment_ids, perm=[1,0]),  # int
        # input_mask=tf.transpose(mask, perm=[1,0])
        input_ids=input_ids,
        seg_ids=segment_ids,
        input_mask=mask
    )
    # print("in create_model, after model： ")
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    # print("inputIds shape:", input_ids.shape[1].value)
    # print("embeeding_shape1",max_seq_length)
    # print("embeeding_shape2", embedding.shape[2].value)

    # 算序列真实长度 加lstm
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    # print("before using crf")
    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    # print("after using crf1")
    rst = blstm_crf.add_blstm_crf_layer(crf_only=False)
    # print("after using crf2")
    return rst

    # output_layer shape is
    # if is_training:
    #     output_layer = tf.keras.layers.Dropout(rate=0.1)(embedding)
    # logits = hidden2tag(embedding,num_labels)
    # # TODO test shape
    # logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    # if FLAGS.crf:
    #     mask2len = tf.reduce_sum(mask,axis=1)
    #     loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
    #     predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
    #     return (loss, logits,predict)
    #
    # else:
    #     loss,predict  = softmax_layer(logits, labels, num_labels, mask)
    #
    #     return (loss, logits, predict)


def model_fn_builder(xlnet_config, num_labels, init_checkpoint, use_tpu):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_masks = features["input_masks"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # print("test1: create_model has error!")
        (total_loss, logits, trans, predicts) = create_model(xlnet_config, is_training, input_ids, input_masks, segment_ids,
                                                             label_ids, num_labels)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars,
                                                                                                          init_checkpoint)
            # tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                         init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, _, _ = model_utils.get_train_op(FLAGS, total_loss)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits, num_labels, input_masks):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels - 1, weights=input_masks)
                return {
                    "confusion_matrix": cm
                }
                #

            eval_metrics = (metric_fn, [label_ids, logits, num_labels, input_masks])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    # if i < len(batch_tokens):

    # print("id2label", id2label)
    # print("prediction",prediction)
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]

    if token != "<pad>" and token != "<cls>" and true_l != "X":

        #
        # if predict == "X" and not predict.startswith(prepro_utils.SPIECE_UNDERLINE):
        #     predict = "O"
        line = "{}\t{}\t{}\n".format(token, true_l, predict)
        wf.write(line)

    # token = batch_tokens[i]

    # print("predic",prediction)


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    # a=0
    with open(output_predict_file, 'w') as wf:
        predictions = []
        for m, pred in enumerate(result):
            predictions.extend(pred["predict"])
            # print("this is pred",pred)

        # for m,batch_token in enumerate(batch_tokens):
        #     print("this is for batch token",batch_token)
        # print("this is batch_tokens_list",batch_tokens)
        #print("in the writer", len(batch_tokens))
        for i, prediction in enumerate(predictions):
            # print("in Writer",prediction)

            _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)
            # a=i



def Writer2(output_predict_file, result, batch_tokens, batch_labels, id2label):
    # a=0
    with open(output_predict_file, 'w') as wf:
        predictions = []
        for m, pred in enumerate(result):
            predictions.extend(pred["predict"])
            # print("this is pred",pred)

        # for m,batch_token in enumerate(batch_tokens):
        #     print("this is for batch token",batch_token)
        # print("this is batch_tokens_list",batch_tokens)
        #print("in the writer", len(batch_tokens))
        for i, prediction in enumerate(predictions):
            # print("in Writer",prediction)

            _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)
            # a=i


class XLNetPredictRecorder(object):
    """Default predict recorder for XLNet"""

    def __init__(self,
                 output_dir,
                 label_list,
                 max_seq_length,
                 tokenizer,
                 predict_tag=None):
        """Construct XLNet predict recorder"""
        self.output_path = os.path.join(output_dir,
                                        "predict.{0}.json".format(predict_tag if predict_tag else str(time.time())))
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def _write_to_json(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:
            json.dump(data_list, file, indent=4)

    def _write_to_text(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:
            for data in data_list:
                file.write("{0}\n".format(data))

    def record(self,
               predicts):
        decoded_results = []
        for predict in predicts:
            input_tokens = self.tokenizer.ids_to_tokens(predict["input_ids"])
            input_masks = predict["input_masks"]
            input_labels = [self.label_list[idx] for idx in predict["label_ids"]]
            output_predicts = [self.label_list[idx] for idx in predict["predict_ids"]]

            decoded_tokens = []
            decoded_labels = []
            decoded_predicts = []
            results = zip(input_tokens, input_masks, input_labels, output_predicts)
            for input_token, input_mask, input_label, output_predict in results:
                if input_token in ["<cls>", "<sep>"] or input_mask == 1:
                    continue

                if output_predict in ["<pad>", "<cls>", "<sep>", "X"]:
                    output_predict = "O"

                if input_token.startswith(prepro_utils.SPIECE_UNDERLINE):
                    decoded_tokens.append(input_token)
                    decoded_labels.append(input_label)
                    decoded_predicts.append(output_predict)
                else:
                    decoded_tokens[-1] = decoded_tokens[-1] + input_token

            decoded_text = "".join(decoded_tokens).replace(prepro_utils.SPIECE_UNDERLINE, " ")
            decoded_label = " ".join(decoded_labels)
            decoded_predict = " ".join(decoded_predicts)

            decoded_result = {
                "text": prepro_utils.printable_text(decoded_text),
                "label": decoded_label,
                "predict": decoded_predict,
            }

            decoded_results.append(decoded_result)

        self._write_to_json(decoded_results, self.output_path)


def main(_):
    #logging.set_verbosity(logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(20)



    processors = {"ner": NerProcessor}
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.xlnet_config_file)
    # if FLAGS.max_seq_length > xlnet_config.max_position_embeddings:
    #     raise ValueError(
    #         "Cannot use sequence length %d because the xlnet model "
    #         "was only trained up to sequence length %d" %
    #         (FLAGS.max_seq_length, xlnet_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    # print("这是processor",processor)
    label_list = processor.get_labels()

    # tokenizer = prepro_utils.FullTokenizer(sp_model_file=FLAGS.vocab_file,
    #                                        lower_case=FLAGS.do_lower_case)
    tokenizer = XLNetTokenizer(
        sp_model_file=FLAGS.vocab_file,
        lower_case=FLAGS.do_lower_case)



    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    example_converter = XLNetExampleConverter(
        label_list=label_list,
        max_seq_length=FLAGS.max_seq_length,
        tokenizer=tokenizer)

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        # 计算tokes数量：++++++++++++++
        alltokenssss=0
        for e in train_examples:
            l=len(tokenizer.tokenize(e.text))
            alltokenssss+=l
        print("nnnnnnnnnnnnnnnnewtraintokens",alltokenssss)

        # +++++++++++++++++++++++++++++++++



        np.random.shuffle(train_examples)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    # model_fn = model_fn_builder(
    #     xlnet_config=xlnet_config,
    #     num_labels=len(label_list),
    #     init_checkpoint=FLAGS.init_checkpoint,
    #     use_tpu=FLAGS.use_tpu)
    model_builder = XLNetModelBuilder(
        model_config=xlnet_config,
        use_tpu=FLAGS.use_tpu)

    model_fn = model_builder.get_model_fn(label_list)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    # tpu_config = model_utils.configure_tpu(FLAGS)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        export_to_tpu=False,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # _, _ = filed_based_convert_examples_to_features(
        #     train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        example_converter.file_based_convert_examples_to_features(train_examples, train_file)


        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d"%len(train_examples))
        tf.logging.info("  Batch size = %d"% FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d"% num_train_steps)

        # train_input_fn = file_based_input_fn_builder(
        #     input_file=train_file,
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=True,
        #     drop_remainder=True)
        train_input_fn = XLNetInputBuilder.get_file_based_input_fn(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True
        )

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        # 计算tokes数量：++++++++++++++
        alldevtokenssss = 0
        for e in eval_examples:
            l = len(tokenizer.tokenize(e.text))
            alldevtokenssss += l
        print("nnnnnnnnnnnnnnnnewdevtokens", alldevtokenssss)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d"% len(eval_examples))
        tf.logging.info("  Batch size = %d"% FLAGS.eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        # eval_features, batch_tokens, batch_labels = example_converter.convert_examples_to_features2return(
        #     eval_examples)

        #dev_input_fn = XLNetInputBuilder.get_input_builder(dev_features, FLAGS.max_seq_length, False, False)
        eval_features = example_converter.convert_examples_to_features(eval_examples)
        eval_input_fn = XLNetInputBuilder.get_input_builder(eval_features, FLAGS.max_seq_length, False, False)
        # eval_input_fn = file_based_input_fn_builder(
        #     input_file=eval_file,
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=False,
        #     drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn)
        #output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

        precision = result["precision"]
        recall = result["recall"]
        f1_score = 2.0 * precision * recall / (precision + recall)


        tf.logging.info("***** Evaluation result *****")
        tf.logging.info("  Precision (token-level) = %s"% str(precision))
        tf.logging.info("  Recall (token-level) = %s"% str(recall))
        tf.logging.info("  F1 score (token-level) = %s"% str(f1_score))

    #
    if FLAGS.do_predict:
        with open(FLAGS.middle_output + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        # 计算tokes数量：++++++++++++++
        alltesttokenssss = 0
        for e in predict_examples:
            l = len(tokenizer.tokenize(e.text))
            alltesttokenssss += l
        print("nnnnnnnnnnnnnnnnewtesttokens", alltesttokenssss)

        # +++++++++++++++++++++++++++++++++
        #np.random.shuffle(predict_examples)

       # predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        predict_features, batch_tokens, batch_labels = example_converter.convert_examples_to_features2return(
            predict_examples)

        #xlnet 写法

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d"% len(predict_examples))
        tf.logging.info("  Batch size = %d"% FLAGS.predict_batch_size)

        predict_input_fn = XLNetInputBuilder.get_input_builder(predict_features, FLAGS.max_seq_length, False, False)

        result = estimator.predict(input_fn=predict_input_fn)

        result1 = estimator.evaluate(input_fn=predict_input_fn)
        # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

        precision = result1["precision"]
        recall = result1["recall"]
        f1_score = 2.0 * precision * recall / (precision + recall)

        tf.logging.info("***** test result *****")
        tf.logging.info("  Precision (token-level) = %s" % str(precision))
        tf.logging.info("  Recall (token-level) = %s" % str(recall))
        tf.logging.info("  F1 score (token-level) = %s" % str(f1_score))


        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # # # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # # # conlleval.pl we  discarding it directly
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label)


    if FLAGS.do_export:
        tf.logging.info("***** Running exporting *****")
        tf.io.gfile.makedirs(FLAGS.export_dir)
        serving_input_fn = XLNetInputBuilder.get_serving_input_fn(FLAGS.max_seq_length)
        estimator.export_saved_model(FLAGS.export_dir, serving_input_fn, as_text=False)




if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("xlnet_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
