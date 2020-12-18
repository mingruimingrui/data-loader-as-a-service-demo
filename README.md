
This repository implements implements data loading as a stand-alone service
for model training. Under the context of multi-gpu training, having data
loading as a service comes with itself a number of rather nifty advantages
and one big disadvantage.

- Makes distributed sampling redundant since there can now be a single souce
of data generation. This also greatly simplifies the code for writing
memory-efficient loading of extremely large datasets.
- Under the context of multiple nodes, we can avoid the need to duplicate the
data on each machine. All our dataset just has to be stored on the machine
hosting the data loader service.

However by making data loading a service, we have essentially created a
coupling between processes (from no-coupling to 1 is alot).
While not necessarily a bad thing, this does make testing, debugging and
development work a little more tedious by requiring one additional prior step.

## Getting Started

### Installation

This package is set up such that it can be ran as a module from the console.
Requirements can be found in [`requirements.txt`](./requirements.txt).

```sh
python -m transformers_pretraining -h
```

Installation is good-to-have but not necessary, you only have to do it
if you wish to be able to run the code from any directory of your choosing.
Installation can be done using pip.

```sh
pip install .
```

### Usage

The following example pre-trains a roberta model from scratch. You will have
to provide your own text files and pre-trained huggingface tokenizer.

The following command starts the data loader service for roberta training.

```sh
python -m transformers_pretraining serve_dataset \
    path_to_train_file_1 path_to_train_file_2 ... \
    --tokenizer path_to_tokenizer \
    --port port_to_host_on
```

It is highly recommended to enable optimal batching to reduce amount of
unnecessary computation.
To enable said feature, simply use the `--do-optimal-batching` option.

```sh
python -m transformers_pretraining serve_dataset \
    path_to_train_file_1 path_to_train_file_2 ... \
    --tokenizer path_to_tokenizer \
    --port port_to_host_on \
    --do-optimal-batching
```

Once the data loader service has started, the training script can be ran.
Examples of model config and training arguments can be found in
[./configs](./configs).

```sh
python -m transformers_pretraining \
    -c path_to_model_config.json \
    -d dataset_loader_host:dataset_loader_port \
    -t path_to_tokenizer \
    --training-args path_to_training_args.json
```

To evaluate on validation data, use the `--eval-data-files` option.

```sh
python -m transformers_pretraining \
    -c path_to_model_config.json \
    -d dataset_loader_host:dataset_loader_port \
    -t path_to_tokenizer \
    --training-args path_to_training_args.json \
    --eval-data-files path_to_validation_file
```

To train on multiple GPUs, use the [torch distributed launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

## Future improvements

- Add a feature to cache batches on inference worker side.
