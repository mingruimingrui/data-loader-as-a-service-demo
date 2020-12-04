import json
import msgpack
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

int64_byte_size = np.dtype(np.int64).itemsize
bool_byte_size = np.dtype(np.bool).itemsize


@dataclass
class BatchForLanguageModeling:
    """An input batch for language modeling training and inference."""

    idxs: np.ndarray
    texts: List[str]
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: Optional[np.ndarray] = None

    def to_bytes(self) -> bytes:
        """Binarize a batch into bytes in an efficient manner.

        I could have used pickle but pickling numpy arrays and tensors
        is extremely inefficient.
        There are better ways to write this but here I want to have the ease of
        interpretation.
        You shouldn't have to to reference anything else.
        """
        batch_size, seq_len = self.input_ids.shape
        texts_data = msgpack.packb(self.texts)
        meta = {
            'shape': [int(batch_size), int(seq_len)],
            'texts_size': len(texts_data),
            'has_label': self.labels is not None,
        }

        data = [
            (json.dumps(meta) + '\n').encode('utf-8'),
            self.idxs.astype(np.int64).tobytes(),
            texts_data,
            self.input_ids.astype(np.int64).tobytes(),
            self.attention_mask.astype(np.bool).tobytes(),
        ]
        if self.labels is not None:
            data.append(self.labels.astype(np.int64).tobytes())

        return b''.join(data)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Decode bytes into a BatchForLanguageModeling.

        I could have used pickle but pickling numpy arrays and tensors
        is extremely inefficient.
        There are better ways to write this but here I want to have the ease of
        interpretation.
        You shouldn't have to to reference anything else.
        """
        offset = data.find(b'\n')
        meta = json.loads(data[:offset])

        batch_size, seq_len = meta['shape']
        tensor_size = batch_size * seq_len
        offset += 1

        idxs = np.frombuffer(data, np.int64, batch_size, offset)
        offset += int64_byte_size * batch_size

        texts = msgpack.unpackb(data[offset:offset + meta['texts_size']])
        offset += meta['texts_size']

        input_ids = np.frombuffer(data, np.int64, tensor_size, offset)
        offset += int64_byte_size * tensor_size

        attention_mask = np.frombuffer(data, np.bool, tensor_size, offset)
        offset += bool_byte_size * tensor_size

        batch = cls(
            idxs=idxs.copy(),
            texts=texts,
            input_ids=input_ids.reshape(batch_size, seq_len).copy(),
            attention_mask=attention_mask.reshape(batch_size, seq_len).copy(),
        )

        if meta['has_label']:
            batch.labels = np.frombuffer(
                data,
                dtype=np.int64,
                count=tensor_size,
                offset=offset,
            ).reshape(batch_size, seq_len).copy()

        return batch


@dataclass
class BatcherForLanguageModeling:
    """For transforming a list of texts into BatchForLanguageModeling."""

    tokenizer: PreTrainedTokenizerBase
    do_optimal_batching: bool = True
    min_seq_len: int = 0
    max_seq_len: int = 0
    max_batch_size: int = 44 * 2
    max_batch_tokens: int = 44 * 128
    mlm: bool = True
    mlm_probability: float = 0.15
    mask_probability: float = 0.8
    random_probability: float = 0.5

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            raise ValueError(
                'This tokenizer does not have a pad token which is necessary '
                'for batching.'
            )
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                'This tokenizer does not have a mask token which is necessary '
                'for masked language modeling.'
            )

    def __call__(self, texts: List[str]) -> List[BatchForLanguageModeling]:
        # Encode texts and sort if needed
        encoded_texts = []
        for idx, text in enumerate(texts):
            encoding = self.tokenizer.encode(text)
            if self.min_seq_len <= len(encoding <= self.max_seq_len):
                encoded_texts.append([idx, text, encoding])
        if self.do_optimal_batching:
            encoded_texts.sort(key=lambda x: len(x[2]))

        # Form batches
        batches = []
        cur_batch = []
        cur_batch_width = 0
        for idx, text, encoding in encoded_texts:
            new_batch_width = max(cur_batch_width, len(encoding))
            new_batch_tokens = new_batch_width * (len(cur_batch) + 1)

            if (
                len(cur_batch) >= self.max_batch_size or
                new_batch_tokens > self.max_batch_tokens
            ):
                batches.append(self._form_batch(cur_batch))
                cur_batch = [(idx, text, encoding)]
                cur_batch_width = len(encoding)

            else:
                cur_batch.append((idx, text, encoding))
                cur_batch_width = new_batch_width

        if len(cur_batch) > 0:
            batches.append(self._form_batch(cur_batch))

        return batches

    def _form_batch(
        self,
        encoded_texts: List[Tuple[int, str, List[int]]]
    ) -> BatchForLanguageModeling:
        longest_seq_len = max(map(lambda x: len(x[2]), encoded_texts))
        idxs = []
        texts = []
        input_ids = np.full(
            (len(encoded_texts), longest_seq_len),
            self.tokenizer.pad_token_id,
            dtype=np.int64
        )
        attention_mask = np.full(input_ids.shape, False, dtype=np.bool)

        for i, (idx, text, encoding) in enumerate(encoded_texts):
            idxs.append(idx)
            texts.append(text)
            input_ids[i, :len(encoding)] = encoding
            attention_mask[i, :len(encoding)] = True

        batch = BatchForLanguageModeling(
            idxs=np.array(idxs, dtype=np.int64),
            texts=texts,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.mlm:
            self._mask_batch(batch)

        return batch

    def _mask_batch(
        self,
        batch: BatchForLanguageModeling
    ) -> BatchForLanguageModeling:
        # Determine locations of labels and initialize
        shape = batch.input_ids.shape
        masked_indices = np.random.rand(*shape) < self.mlm_probability
        masked_indices[~batch.attention_mask] = False
        special_token_indices = [self.tokenizer.get_special_tokens_mask(
            encoding,
            already_has_special_tokens=True,
        ) for encoding in batch.input_ids.tolist()]
        special_token_indices = np.array(special_token_indices, dtype=np.bool)
        masked_indices[special_token_indices] = False

        batch.labels = batch.input_ids.copy()
        batch.labels[~masked_indices] = -100

        # In {mask_probability} of time, mask input tokens
        indices_replaced = np.random.rand(*shape) <= self.mask_probability
        indices_replaced = indices_replaced & masked_indices
        batch.input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # In {mask_probability} * {random_probability} of time,
        # replace with random tokens
        indices_random = np.random.rand(*shape) <= self.random_probability
        indices_random = indices_random & masked_indices & ~indices_replaced
        random_tokens = np.random.randint(
            0, len(self.tokenizer),
            size=shape, dtype=np.int64
        )
        batch.input_ids[indices_random] = random_tokens[indices_random]

        return batch
