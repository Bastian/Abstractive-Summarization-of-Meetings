# Taken from https://github.com/asyml/texar/blob/master/examples/bert/config_data.py and modified
max_seq_length = 96
num_train_data = 1731

tfrecord_data_dir = "data"

train_batch_size = 28
max_train_epoch = 250
display_steps = 10  # Print training loss every display_steps; -1 to disable
eval_steps = 75  # Eval on the dev set every eval_steps; -1 to disable

eval_batch_size = 28
test_batch_size = 28

max_decoding_length = 96

feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "src_input_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "src_input_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "src_segment_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "tgt_input_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "tgt_labels": ["tf.int64", "FixedLenFeature", max_seq_length]
}

feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "src_input_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "src_input_ids": "tf.int32",
    "src_segment_ids": "tf.int32",
    "tgt_input_ids": "tf.int32",
    "tgt_labels": "tf.int32"
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/train.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 100
}

eval_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/eval.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}

test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/test.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}