# HuggingFace Sequence Classification on GCP

<!--ts-->
   * [HuggingFace Sequence Classification on GCP](#huggingface-sequence-classification-on-gcp)
      * [Overview](#overview)
      * [Local Development](#local-development)
      * [Dockerfile](#dockerfile)
      * [Building](#building)
      * [Registering](#registering)
      * [Running](#running)
         * [Google Cloud AI Platform Jobs](#google-cloud-ai-platform-jobs)
         * [Local](#local)
      * [Task Arguments](#task-arguments)
         * [Training / Development Data](#training--development-data)
         * [Other arguments](#other-arguments)
      * [Sequence Length Functions](#sequence-length-functions)
         * [Head / Tail](#head--tail)
         * [Random Span Non-overlapping](#random-span-non-overlapping)
         * [Random Span Overlapping](#random-span-overlapping)
         * [Random Spans "Delete"](#random-spans-delete)
      * [Next Steps](#next-steps)

<!-- Added by: untitled, at: Fri Feb  7 14:04:47 EST 2020 -->

<!--te-->

## Overview

Leveraging the huggingface sequence classification models to train sequence
classifiers in Google Cloud Platform. 


## Local Development

```
virtualenv -p python3 venv
source venv/bin/activate

pip install -r requirements.txt
pip install torch
```


## Dockerfile

* Installs cuda / pytorch
* Installs Apex (omg this was annoying)
* Copies over some python files in this directory

## Building

```
➜  docker build -t IMAGE_NAME .
```

## Registering

```
➜  ./bin/register-image.sh
```

## Running

### Google Cloud AI Platform Jobs

```
➜  ./bin/submit-aiplatform.sh
```

This will automatically stream the logs. If the `bucket` arg is configured, 
which it should be for running in ml-engine. Then you can view the tensorboard logs by running:

```
➜ tensorboard --logdir=YOUR_BUCKET
```

### Local

```
➜ ./bin/run_local.sh
```

## Task Arguments 

### Training / Development Data

The tasks expect data in the format:

| ID | ALPHA_COL | LABEL | TEXT_A | TEXT_B |
|----|-----------|-------|--------|--------|

You'll a split of training / dev files. To format the files correctly, 
see the `bert-utils` section. 

The path to this directory is the argument for `--data_dir`. 

When running the tasks in Google Cloud, if the argument for `data_dir` is a Google Cloud Storage bucket then everything will be copied from that bucket storage path to the training instance.

### Other arguments

You will also need to specify the following: 
* `model_type`: model name according to huggingface; i.e. `bert`, `roberta`, etc.
* `model_name`: type of model via model name, i.e. `bert-large-uncased`
* `task_name`: the name of this particular training or testing task
* `output_dir`: the directory for task output, i.e. `./local`
* `max_seq_length`: the maximum length of a sequence of text to consider for classification, huggingface models max out at 512
* `seq_len_func`: Sequence length function to use. See below for options.
* `seq_func_params`: Sequence length function params. See below for options and examples. 

## Sequence Length Functions

Most BERT models were trained on a maximum sequence length of 512. For this reason, it isn't possible to fine-tune these models on larger length sequences. This repo implements some potential work arounds for documents > 512 which involve selecting different pieces of the document based on the sequencing function and parameters. The various sequence function generators can be found in `features.py`

| Sequence Function           | Function Argument | Arguments                                              | Example Arguments |
|-----------------------------|-------------------|--------------------------------------------------------|-------------------|
| Head / Tail                 | ht                |  (head_len, tail_len). Where head_len + tail_len < 512 | (128, 382)        |
| Random Span Non-Overlapping | rs                | span_length                                            | 16                |
| Random Span Delete          | rsd               | span_length                                            | 16                |
| Head                        | head              | max_seq_length                                         | 512               |
| Random Span Overlapping     | rso               | span_length                                            | 16                |

### Head / Tail

Based on the following paper ["How to Fine-Tune BERT for Text Classification?"](https://arxiv.org/pdf/1905.05583.pdf) the head/tail method selects sequences up to length X from the head of the document and up to length Y from the end of the document. 

### Random Span Non-overlapping

This function splits the sequence into X spans by scrolling through the sequence in order. The function selects Y random spans here without repetition. This function takes as input the sequence length that the Y random spans should be.

### Random Span Overlapping

Similar to the above random spam function however this one will allow for individual token overlap in the spans. 


### Random Spans "Delete"

Selects spans randomly of input length X from the input sequence. After selecting a span it is removed from the sequence when selecting the next. There is no overlap here however this may obstruct the word order specific BERT features. 

## Next Steps

* Naming tensorboard event files based on task name
* Optional knockknock slack updates
* Copy over config files into storage
