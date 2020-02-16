# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2020 Bastian Oppermann
# The original, unmodified file(s) can be found at
# https://github.com/asyml/texar/blob/413e07f859acbbee979f274b52942edd57b335c1/examples/bert/prepare_data.py
"""Produces TFRecord files and modifies data configuration file
"""

import os
import texar.tf as tx

import config_model

from utils import data_utils

MAX_SEQ_LENGTH = 96


def _modify_config_data(max_seq_length, num_train_data):
    # Modify the data configuration file
    config_data_exists = os.path.isfile('./config_data.py')
    if config_data_exists:
        with open("./config_data.py", 'r') as file:
            filedata = file.read()
            filedata_lines = filedata.split('\n')
            idx = 0
            while True:
                if idx >= len(filedata_lines):
                    break
                line = filedata_lines[idx]
                if (line.startswith('num_classes =') or
                        line.startswith('num_train_data =') or
                        line.startswith('max_seq_length =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "num_train_data", num_train_data))
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "max_seq_length", max_seq_length))

        with open("./config_data.py", 'w') as file:
            file.write('\n'.join(filedata_lines))
        print("config_data.py has been updated")
    else:
        print("config_data.py cannot be found")

    print("Data preparation finished")


def main():
    """Prepares data.
    """
    # Loads data
    print("Loading data")

    data_dir = './data'

    tfrecord_output_dir = data_dir
    tx.utils.maybe_create_dir(tfrecord_output_dir)

    processor = data_utils.TsvProcessor()

    num_train_data = len(processor.get_train_examples(data_dir))
    print('num_train_data:%d' % num_train_data)

    tokenizer = tx.data.BERTTokenizer(pretrained_model_name=config_model.bert['pretrained_model_name'])

    # Produces TFRecord files
    data_utils.prepare_TFRecord_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        output_dir=tfrecord_output_dir)

    _modify_config_data(MAX_SEQ_LENGTH, num_train_data)


if __name__ == "__main__":
    main()
