# Abstractive Summarization of Meetings

### IMPORTANT: This repository is work in progress!

This project contains the source code for my bachelor's thesis "Abstractive Summarization of Meetings".

## Requirements

This project was only tested with Python 3.6 but should also work with more recent version of Python.
For dependency versions, take a look at the `requirements.txt` file.

## Execution

### Preparing the data

```
python prepare_data.py
```

reads the `ami.tsv` file and generates 3 TFRecord data files `train.tf_record`, `eval.tf_record`, and `test.tf_record`.
These files are used for training.

### Training

```
python main.py --run_mode=train_and_evaluate
```

starts the training.

## Credits

### Data

The data from `ami.tsv` is taken from the [AMI Corpus](http://groups.inf.ed.ac.uk/ami/corpus/) and processed
using the [NITE XML Toolkit](http://groups.inf.ed.ac.uk/nxt/index.shtml). The code that parses the corpus is
not yet published, but will be soon.

#### License

The AMI Corpus license can be found here: [AMI Meeting Corpus License](http://groups.inf.ed.ac.uk/ami/corpus/license.shtml).

### Code

Main parts of the code are taken from the Texar examples for BERT and Transformers. They can be found under
the following links:

* [BERT example](https://github.com/asyml/texar/blob/413e07f859acbbee979f274b52942edd57b335c1/examples/bert/)
* [Transformer example](https://github.com/asyml/texar/blob/413e07f859acbbee979f274b52942edd57b335c1/examples/transformer/)

These examples are licensed under the [Apache License 2.0](https://github.com/asyml/texar/blob/413e07f859acbbee979f274b52942edd57b335c1/LICENSE#).
Copied files contain a link to their original version in the file header. Any of my modifications
are also licensed under the same license.

### Inspiration

This project was inspired by the GitHub repository [Abstractive Summarization With Transfer Learning](https://github.com/santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning).
This project uses no source code of the repository, though. The repository is also based on the Texar examples and thus
has similar code.

## License

This project is licensed under the [Apache License 2.0](/LICENSE).