import argparse
import atexit
import multiprocessing
import random
import threading
import time
from typing import List, Union

import msgpack
from fastapi import FastAPI, Response
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers_pretraining.batching import Batch, BatcherForLanguageModeling
from transformers_pretraining.cached_generator import CachedGenerator
from transformers_pretraining.io_utils import load_texts_endless
from uvicorn.main import run as run_app

# Worker global variables
args: argparse.Namespace
bacher: BatcherForLanguageModeling


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

    BatcherForLanguageModeling.add_options(parser)

    return parser


def visualize_batch(
    batch: Batch,
    tokenizer: PreTrainedTokenizerFast,
    n: int = 1
):
    entry_id = 0
    sample_text = tokenizer.decode(batch.input_ids[entry_id])
    print(f'Sample Entry: {sample_text}')
    print(f'text: {batch.texts[entry_id]}')
    print(f'input_ids: {batch.input_ids[entry_id]}')
    print(f'attention_mask: {batch.attention_mask[entry_id]}')
    print(f'labels: {batch.labels[entry_id]}')


def worker_init(_args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast):
    """We want to cache the batcher for workers to use later on."""
    global args, batcher
    args = _args
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
    """Each worker receives a chunk of texts to form into batches."""
    texts: List[str] = msgpack.unpackb(data)
    if args.shuffle:
        random.shuffle(texts)
    batches = [batch for batch in batcher(texts)]
    if args.shuffle:
        random.shuffle(batches)
    return msgpack.packb([batch.to_bytes() for batch in batches])


def init_cached_batches(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerFast
):
    """This function creates a generator that generates Batch objects.

    In addition to generating batches from text data,
    this implementation also
    - offload the heavy lifting of text processing to subprocesses
    - limit the amount of memory consumed by using semaphore to ensure
      a maximum amount of lines are loaded into memory at once
    - perform graceful shutdown by resolving subprocesses and other misc
      objects manually
    """
    max_semaphore_value = args.num_workers * 2
    semaphore = threading.Semaphore(max_semaphore_value)
    shutdown_event = threading.Event()

    pool = multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=[args, tokenizer]
    )

    def make_chunk_generator():
        text_generator = load_texts_endless(args.data_files)
        while not shutdown_event.is_set():
            if semaphore.acquire(blocking=True, timeout=0.05):
                chunk = [next(text_generator) for _ in range(args.chunk_size)]
                yield msgpack.packb(chunk)

    data_generator = pool.imap(worker_fn, make_chunk_generator())

    # Oh man why did I choose to use imap?
    # The handling of exit condition is really messy...
    # Currently this function here manually
    # - stops the input of new data
    # - empties out current imap queue
    # - shutdown workers
    # Do let me know if there's a better way to do graceful shutdown
    # that I wasn't aware of.
    def gracefully_shutdown_imap():
        shutdown_event.set()
        while semaphore._value < max_semaphore_value:
            data_generator.next()
            semaphore.release()
        pool.close()
        pool.join()

    atexit.register(gracefully_shutdown_imap)

    def make_batch_generator():
        for data in data_generator:
            semaphore.release()
            # Unwrapped from `yield from msgpack.unpackb(data)` for readability
            for encoded_batch in msgpack.unpackb(data):
                yield encoded_batch

    batch_generator = make_batch_generator()
    cached_batches = CachedGenerator(
        batch_generator,
        cache_size=args.cache_size
    )

    return cached_batches


def make_app(args: argparse.Namespace):
    app = FastAPI()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    # API global variables
    cached_batches: CachedGenerator = None

    @app.get('/')
    def get_sample():
        assert cached_batches is not None
        data = next(cached_batches)
        return Response(content=data)

    @app.on_event('startup')
    def on_startup_event():
        nonlocal cached_batches
        cached_batches = init_cached_batches(args, tokenizer)

        # Await on first entry and visualize.
        while cached_batches.qsize() == 0:
            time.sleep(0.05)
        batch = Batch.from_bytes(cached_batches.peek())
        visualize_batch(batch, tokenizer, 1)

        print(f'Data loading service started on localhost:{args.port}')

    @app.on_event('shutdown')
    def on_shutdown_event():
        print('Gracefully shutting down data loading service')
        if cached_batches is not None:
            cached_batches.close()

    return app, on_shutdown_event


def main(args: Union[argparse.Namespace, List[str], None] = None):
    if not isinstance(args, argparse.Namespace):
        parser = argparse.ArgumentParser(
            description=__doc__,
            conflict_handler='resolve',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        add_options(parser)
        args = parser.parse_args(args)

    app, on_shutdown_event = make_app(args)
    run_app(app, port=args.port, log_level='warning')
    on_shutdown_event()


if __name__ == "__main__":
    main()
