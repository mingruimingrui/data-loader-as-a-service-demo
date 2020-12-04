"""For various utility functions."""

import threading
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue


def enqueue(
    context: threading.Event,
    queue: Queue,
    item: object,
    timeout: float = 0.05
):
    while not context.is_set():
        try:
            return queue.put(item, block=True, timeout=timeout)
        except QueueFull:
            continue


def dequeue(
    context: threading.Event,
    queue: Queue,
    timeout: float = 0.05
) -> object:
    while not context.is_set():
        try:
            return queue.get(block=True, timeout=timeout)
        except QueueEmpty:
            continue
