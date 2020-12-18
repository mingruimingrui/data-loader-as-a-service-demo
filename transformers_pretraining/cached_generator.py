from types import GeneratorType
from queue import Queue
import threading
from transformers_pretraining.queue_utils import enqueue, dequeue


class CachedGenerator:
    """A wrapper around a generator which caches the output.

    Caching outputs allow the next entry to be available immediately on calling
    __next__.
    The problem with this wrapper is that it's not exactly efficient.
    The expected use cases are mostly for generators with low generation rates
    (< 10k/s) and even lower consumption rates (< 3k/s).

    This class is also not written in a thread safe manner.
    """

    def __init__(self, generator: GeneratorType, cache_size: int = 2):
        self.generator = generator
        self.cache = Queue(maxsize=cache_size)

        self.done_event = threading.Event()
        self.shutdown_event = threading.Event()

        def fill_cache():
            for data in generator:
                if self.shutdown_event.is_set():
                    return
                enqueue(self.shutdown_event, self.cache, data)
            self.done_event.set()

        self.worker = threading.Thread(target=fill_cache)
        self.worker.setDaemon(True)
        self.worker.start()

    def qsize(self):
        return self.cache.qsize()

    def peek(self):
        if self.qsize() == 0:
            raise RuntimeError()
        return self.cache.queue[0]

    def close(self):
        self.shutdown_event.set()
        self.worker.join()

    def __del__(self):
        self.close()

    def __next__(self):
        if self.done_event.is_set() and self.cache.qsize() == 0:
            raise StopIteration()
        return dequeue(self.shutdown_event, self.cache)

    def __iter__(self):
        return self
