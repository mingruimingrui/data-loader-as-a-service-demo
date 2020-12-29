
This repository implements implements data loading as a stand-alone service
for model training. Under the context of multi-gpu training, having data
loading as a service comes with itself a number of rather nifty advantages
and one big disadvantage.

- Makes distributed sampling redundant since there can now be a single souce
  of truth. This also greatly simplifies the code for writing
  memory-efficient loading of extremely large datasets.
- Under the context of multiple nodes, we can avoid the need to duplicate the
  data on each machine. All our dataset just has to be stored on the machine
  hosting the data loader service.

However by making data loading a service, we have essentially created a
coupling between processes (from no-coupling to 1 is alot).
While not necessarily a bad thing, this does make testing, debugging and
development work a little more tedious by requiring one additional prior step.

Another consideration would possibly be networking.
With data coming from just a single machine, combined with the relatively large
size of data, overloading of network bandwith is a very realistic possibility.

## File Structure

```
transformers_pretraining
├── __init__.py
├── __main__.py             # Entrypoint to module (contains the main function)
├── bin
│   ├── __init__.py
│   ├── serve_dataset.py    # Script to start data loader service
│   └── train_roberta.py    # Script to train roberta model
├── batching.py             # Utility for forming text into batches
├── cached_generator.py     # Utility for caching generator output
├── io_utils.py             # Utility for doing I/O
├── queue_utils.py          # Utility for working with queues
└── trainers.py             # Custom classes to perform model training
```

## Getting Started

### Requirements

Requirements can be found in [`requirements.txt`](./requirements.txt).

### Installation (is completely optional)

Because the code is set up such that it can be ran as a module.

```
python -m transformers_pretraining -h
```

You don't have to install the package at all unless you wish to be able to
run the code from any directory of your choosing.

## Usage

### Running the data loader service

The following command starts the data loader service for roberta training.
You will have to provide your own text files and pre-trained huggingface
tokenizer.

```sh
python -m transformers_pretraining serve_dataset \
    path_to_train_file_1 path_to_train_file_2 ... \
    --tokenizer path_to_tokenizer \
    --port port_to_host_on \
    --do-optimal-batching
```

### Interpreting results from the dataloader service

To keep things minimal, the dataloader service has only one method, `GET /`.

The following script shows how the output from `GET /` can be used.

```python
import requests
from transformers_pretraining.batching import Batch

resp = requests.get('http://localhost:5678')
batch = Batch.from_bytes(resp.content)
# Where batch has the following fields of interest
# - texts (List[str]): The original texts used to form the batch
# - idxs (Array[int] [batch_size]): Original position of each text
# - input_ids (Array[int] [batch_size, seq_len]): texts but formed into a 2d
#     tensor ready for model input.
# - attention_mask (Array[int] [batch_size, seq_len]): Attention mask for
#     position of non-special tokens.
# - labels (Array[int] [batch_size, seq_len]): Target labels for masked
#     language modeling.

# To perform inference with the batch
model_output = roberta_model(**batch.to_model_inputs(device))
```

### Running the model training process

The following command can be used to start roberta training.
Examples of `model_config` and `training_args` can be found in
[./configs](./configs).

```sh
python -m transformers_pretraining \
    -c path_to_model_config.json \
    -d dataset_loader_host:dataset_loader_port \
    -t path_to_tokenizer \
    --training-args path_to_training_args.json
```

To train on multiple GPUs, use the [torch distributed launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).
