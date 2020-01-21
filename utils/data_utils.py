# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2020 Bastian Oppermann
# The original, unmodified file(s) can be found at
# https://github.com/asyml/texar/blob/413e07f859acbbee979f274b52942edd57b335c1/examples/bert/utils/data_utils.py
import os
import csv
import collections

import tensorflow as tf

import texar.tf as tx

pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3


class InputExample:
    """A single training/test example for text summarization."""

    def __init__(self, guid, src_text, tgt_text=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            src_text: string. The untokenized source text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            tgt_text: (Optional) string. The target of the example text. This should be
                specified for train, dev and test examples, but not for predict examples.
        """
        self.guid = guid
        self.src_text = src_text
        self.tgt_text = tgt_text


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, src_input_ids, src_segment_ids, tgt_input_ids=None, tgt_labels=None):
        self.src_input_ids = src_input_ids
        self.src_segment_ids = src_segment_ids

        self.tgt_input_ids = tgt_input_ids
        self.tgt_labels = tgt_labels


class DataProcessor(object):
    """Base class for data converters."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines


class TsvProcessor(DataProcessor):
    """Processor for tsv files."""

    @staticmethod
    def __get_inputs(lines, data_type):
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                continue
            guid = "data-%s-%d" % (data_type, i)
            src_text = tx.utils.compat_as_text(line[0])
            tgt_text = tx.utils.compat_as_text(line[1])
            example = InputExample(guid=guid, src_text=src_text, tgt_text=tgt_text)
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "data.train.tsv"))
        return self.__get_inputs(lines=lines, data_type='train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "data.dev.tsv"))
        return self.__get_inputs(lines=lines, data_type='dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "data.test.tsv"))
        return self.__get_inputs(lines=lines, data_type='test')


class PredictProcessor:

    def __read_txt(cls, input_file):
        """Reads a text file."""
        with tf.gfile.Open(input_file, "r") as f:
            return f.readlines()

    def get_examples(self, data_dir):
        lines = self.__read_txt(os.path.join(data_dir, "predict.txt"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "data-predict-%d" % i
            src_text = tx.utils.compat_as_text(line)
            examples.append(InputExample(guid=guid, src_text=src_text))

        return examples


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    # noinspection PyProtectedMember
    sep_token_id = tokenizer._map_token_to_id('[SEP]')

    src_input_ids, src_segment_ids, _ = tokenizer.encode_text(text_a=example.src_text, max_seq_length=max_seq_length)

    if example.tgt_text is not None:
        tgt_input_ids, _, _ = tokenizer.encode_text(text_a=example.tgt_text, max_seq_length=max_seq_length)

        tgt_input_ids[0] = bos_token_id  # Replace [CLS] token with bos token
        if sep_token_id in tgt_input_ids:
            tgt_input_ids[tgt_input_ids.index(sep_token_id)] = eos_token_id  # Replace [SEP] token with eos token
        else:
            tgt_input_ids[len(tgt_input_ids) - 1] = eos_token_id  # Replace last token with eos token

        # The target labels are the same as the decoder inputs, but shifted by one
        tgt_labels = tgt_input_ids[1:]
        tgt_labels.append(0)

        feature = InputFeatures(src_input_ids=src_input_ids,
                                src_segment_ids=src_segment_ids,
                                tgt_input_ids=tgt_input_ids,
                                tgt_labels=tgt_labels)
    else:
        feature = InputFeatures(src_input_ids=src_input_ids,
                                src_segment_ids=src_segment_ids)

    return feature


def convert_examples_to_features_and_output_to_files(examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["src_input_ids"] = create_int_feature(feature.src_input_ids)
        features["src_segment_ids"] = create_int_feature(feature.src_segment_ids)

        if feature.tgt_input_ids is not None:
            features["tgt_input_ids"] = create_int_feature(feature.tgt_input_ids)
            features["tgt_labels"] = create_int_feature(feature.tgt_labels)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def prepare_TFRecord_data(processor, tokenizer, data_dir, max_seq_length, output_dir):
    """
    Args:
        processor: Data Preprocessor, which must have get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord in.
    """

    train_examples = processor.get_train_examples(data_dir)
    train_file = os.path.join(output_dir, "train.tf_record")
    convert_examples_to_features_and_output_to_files(train_examples, max_seq_length, tokenizer, train_file)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(output_dir, "eval.tf_record")
    convert_examples_to_features_and_output_to_files(eval_examples, max_seq_length, tokenizer, eval_file)

    test_examples = processor.get_test_examples(data_dir)
    test_file = os.path.join(output_dir, "test.tf_record")
    convert_examples_to_features_and_output_to_files(test_examples, max_seq_length, tokenizer, test_file)
