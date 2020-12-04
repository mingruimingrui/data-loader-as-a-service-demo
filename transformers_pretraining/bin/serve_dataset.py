"""Start the data loader service."""

import argparse
import multiprocessing
import os
import threading
from queue import Queue
from typing import List

import msgpack
from fastapi import FastAPI, Response
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers_pretraining.batching import (BatcherForLanguageModeling,
                                               BatchForLanguageModeling)
from transformers_pretraining.io_utils import load_texts_endless
from transformers_pretraining.queue_utils import dequeue, enqueue
from uvicorn.main import run as run_app

shutdown_event: threading.Event
batch_queue: Queue
dataloader_process: threading.Thread
bacher: BatcherForLanguageModeling


def add_batching_options(
    parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    # Batching configs
    batching_group = parser.add_argument_group('Batching Options')
    batching_group.add_argument(
        '--do-optimal-batching', action='store_true',
        help='Batch by sequence length to minimize padding token.')
    batching_group.add_argument(
        '--min-seq-len', type=int, default=1,
        help='Minimum sequence length.')
    batching_group.add_argument(
        '--max-seq-len', type=int, default=128,
        help='Maximum sequence length.')
    batching_group.add_argument(
        '--max-batch-size', type=int, default=88,
        help='Maximum number of entries per batch.')
    batching_group.add_argument(
        '--max-batch-tokens', type=int, default=int(88 * 128 / 8),
        help='Maximum number of tokens per batch.')

    # Augmentation configs
    augmentation_group = parser.add_argument_group('Augmentation Options')
    augmentation_group.add_argument(
        '--mlm-probability', type=float, default=0.15,
        help='Probability of token masked.')
    augmentation_group.add_argument(
        '--mask-probability', type=float, default=0.8,
        help='Probability of replacing with mask token.')
    augmentation_group.add_argument(
        '--random-probability', type=float, default=0.5,
        help='Probability of replacing with random token.')

    return parser


def add_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        'data_files', nargs='+',
        help='List of text files to generate dataset.')
    parser.add_argument(
        '-t', '--tokenizer-path', required=True,
        help='Path to PretrainedTokenizer.')
    parser.add_argument(
        '-p', '--port', type=int, default=5678,
        help='Port number to host data loader service.')
    parser.add_argument(
        '-j', '--num-workers', type=int, default=2,
        help='Number of workers to use to tokenize data.')

    # Data loading configs
    data_loading_group = parser.add_argument_group('Data Loading Options')
    data_loading_group.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle order of batches?')
    data_loading_group.add_argument(
        '--chunk-size', type=int, default=8192,
        help='Chunk size to read raw text in.')
    data_loading_group.add_argument(
        '--cache-size', type=int, default=100,
        help='Cache used to store completed batches.')

    add_batching_options(parser)

    return parser


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def worker_init(args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast):
    """We want to cache the batcher for workers to use later on."""
    global batcher
    batcher = BatcherForLanguageModeling(
        tokenizer=tokenizer,
        do_optimal_batching=args.do_optimal_batching,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        mlm=True,
        mlm_probability=args.mlm_probability,
        mask_probability=args.mask_probability,
        random_probability=args.random_probability,
    )


def worker_fn(data: bytes) -> bytes:
    texts: List[str] = msgpack.unpackb(data)
    return msgpack.packb([batch.to_bytes() for batch in batcher(texts)])


def start_data_loader_process(
    args: argparse.Namespace,
    batch_queue: Queue,
    shutdown_event: threading.Event,
):
    semaphore = threading.Semaphore(args.num_workers * 2)
    tokenizer = load_tokenizer(args.tokenizer_path)

    def make_chunk_generator():
        text_generator = load_texts_endless(args.data_files)
        while not shutdown_event.is_set():
            if (
                not shutdown_event.is_set() and
                semaphore.acquire(blocking=True, timeout=0.05)
            ):
                chunk = [next(text_generator) for _ in range(args.chunk_size)]
                yield msgpack.packb(chunk)

    with multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=[args, tokenizer]
    ) as pool:
        data_generator = pool.imap(worker_fn, make_chunk_generator())
        while True:
            if shutdown_event.is_set():
                pool.close()
                return print('Shutdown data loader')

            try:
                data = data_generator.next(timeout=0.05)
            except multiprocessing.context.TimeoutError:
                continue
            except Exception as e:
                shutdown_event.set()
                raise e

            semaphore.release()
            for encoded_batch in msgpack.unpackb(data):
                enqueue(shutdown_event, batch_queue, encoded_batch)

    print('Bruh')
    shutdown_event.set()


def main(args: argparse.Namespace):
    app = FastAPI()

    @app.get('/')
    def get_sample():
        data = dequeue(shutdown_event, batch_queue)
        return Response(content=data)

    @app.on_event('startup')
    def on_startup_event():
        global shutdown_event, batch_queue, dataloader_process

        shutdown_event = threading.Event()

        # We validate that the tokenizer can be loaded in main thread
        tokenizer = load_tokenizer(args.tokenizer_path)

        # Also ensure that all data files exists
        for filepath in args.data_files:
            assert os.path.isfile(filepath)

        batch_queue = Queue(args.cache_size)
        dataloader_process = threading.Thread(
            target=start_data_loader_process,
            args=[args, batch_queue, shutdown_event],
        )
        dataloader_process.setDaemon(True)
        dataloader_process.start()

        resp = get_sample()
        if shutdown_event.is_set():
            on_shutdown_event()

        batch = BatchForLanguageModeling.from_bytes(resp.body)

        entry_id = 0
        sample_text = tokenizer.decode(batch.input_ids[entry_id])
        print(f'Sample Entry: {sample_text}')
        print(f'text: {batch.texts[entry_id]}')
        print(f'input_ids: {batch.input_ids[entry_id]}')
        print(f'attention_mask: {batch.attention_mask[entry_id]}')
        print(f'labels: {batch.labels[entry_id]}')

        print(f'Dataset service started on localhost:{args.port}')

    @app.on_event('shutdown')
    def on_shutdown_event():
        print('Shutting down dataset service')

        shutdown_event.set()
        dataloader_process.join()

    run_app(app, port=args.port, log_level='warning')
    on_shutdown_event()
