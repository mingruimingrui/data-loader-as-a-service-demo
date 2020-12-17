import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import msgpack
import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

int64_byte_size = np.dtype(np.int64).itemsize
bool_byte_size = np.dtype(np.bool).itemsize


@dataclass
class Batch:
    """An input batch for transformer input."""

    texts: List[str] = field(metadata={'help': 'List of original texts'})
    idxs: np.ndarray = field(metadata={'help': 'Position of text'})
    input_ids: np.ndarray = field(metadata={
        'help': 'texts formed into input tensor'
    })
    attention_mask: np.ndarray = field(metadata={
        'help': 'Attention mask for position of non-special tokens'
    })
    labels: Optional[np.ndarray] = field(default=None, metadata={
        'help': 'Prediction labels for model training.'
    })

    __model_input_types__ = {
        'input_ids': torch.LongTensor,
        'attention_mask': torch.FloatTensor,
        'labels': torch.LongTensor,
    }

    def to_model_inputs(
        self,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Converts batch into a dictionary ready for use in model."""
        return {
            dtype(getattr(self, name)).to(device)
            for name, dtype in self.__model_input_types__.items()
            if getattr(self, name) is not None
        }

    def to_bytes(self) -> bytes:
        """Binarize a batch into bytes in an efficient manner."""
        metas = []
        datas = []
        for name, value in vars(self).items():
            # Attributes beginning with _ are assumed to be static variables
            if name.startswith('_'):
                continue

            # Also skip if field is empty
            if value is None:
                continue

            # We binarizae arrays and other data types differently
            if isinstance(value, np.ndarray):
                data = value.tobytes()
                metas.append({
                    'name': name,
                    'is_array': True,
                    'size': len(data),
                    'dtype': value.dtype.name,
                    'shape': value.shape,
                    'count': value.size,
                })
                datas.append(data)

            else:
                # Does msgpack always work? Probably not but we only have
                # to handle texts at the moment.
                data = msgpack.packb(value)
                metas.append({
                    'name': name,
                    'is_array': False,
                    'size': len(data),
                })
                datas.append(data)

        # Binarize metas and prepend to datas
        datas = [(json.dumps(metas) + '\n').encode()] + datas
        return b''.join(datas)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Decode bytes into a Batch."""
        offset = data.find(b'\n')
        metas = json.loads(data[:offset])
        offset += 1

        kwargs = {}
        for meta in metas:
            if meta['is_array']:
                value = np.frombuffer(
                    data,
                    dtype=meta['dtype'],
                    count=meta['count'],
                    offset=offset
                ).reshape(meta['shape']).copy()

            else:
                value = msgpack.unpackb(data[offset:offset + meta['size']])

            kwargs[meta['name']] = value
            offset += meta['size']

        return cls(**kwargs)


@dataclass
class BatcherForLanguageModeling:
    """For transforming a list of texts into Batch."""

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

    @staticmethod
    def add_options(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            '-t', '--tokenizer-path', required=True,
            help='Path to saved PretrainedTokenizer.')

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

    @classmethod
    def from_args(cls, args: argparse.Namespace, **kwargs):
        return BatcherForLanguageModeling(
            PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path),
            do_optimal_batching=args.do_optimal_batching,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            max_batch_tokens=args.max_batch_tokens,
            mlm_probability=args.mlm_probability,
            mask_probability=args.mask_probability,
            random_probability=args.random_probability,
            **kwargs,
        )

    def __call__(self, texts: List[str]) -> List[Batch]:
        # Encode texts and sort if needed
        encoded_texts = []
        for idx, text in enumerate(texts):
            encoding = self.tokenizer.encode(text)
            if self.min_seq_len <= len(encoding) <= self.max_seq_len:
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
    ) -> Batch:
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

        batch = Batch(
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
        batch: Batch
    ) -> Batch:
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
