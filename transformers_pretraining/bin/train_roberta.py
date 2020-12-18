"""Pre-train roberta model."""

import argparse
import json
from typing import List, Optional, Tuple

import torch
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.training_args import TrainingArguments
from transformers_pretraining.batching import Batch, BatcherForLanguageModeling
from transformers_pretraining.trainer import TrainerForMaskedLM


def add_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '-c', '--model-config', required=True,
        help='Path to model config file.')
    parser.add_argument(
        '-t', '--tokenizer-path', required=True,
        help='Path to saved PretrainedTokenizer.')
    parser.add_argument(
        '-d', '--dataset-host', required=True,
        help='Url to dataset service.')
    parser.add_argument(
        '--training-args', required=True,
        help='Path to training arguments file.')
    parser.add_argument(
        '--eval-data-files', nargs='+',
        help='List of text files to generate dataset.')
    BatcherForLanguageModeling.add_options(parser)
    return parser


def load_training_arguments(
    local_rank: int = -1,
    filepath: Optional[str] = None
) -> TrainingArguments:
    train_args_dict = {
        'output_dir': 'checkpoints',
        'gradient_accumulation_steps': 768,
        'learning_rate': 6e-3,
        'warmup_steps': 2000,
        'max_steps': 10000,
        'logging_steps': 1,
        'eval_steps': 100,
        'save_steps': 100,
        'save_total_limit': 10,
    }
    if filepath is not None:
        with open(filepath, 'r') as f:
            train_args_dict.update(json.load(f))
    train_args_dict['local_rank'] = local_rank
    return TrainingArguments(**train_args_dict)


def load_tokenizer_and_model(
    tokenizer_path: str,
    model_config_path: str,
) -> Tuple[PreTrainedTokenizerFast, RobertaForMaskedLM]:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    with open(model_config_path) as f:
        model_config = json.load(f)
    model_config['vocab_size'] = len(tokenizer)
    model_config['bos_token_id'] = tokenizer.cls_token_id
    model_config['pad_token_id'] = tokenizer.pad_token_id
    model_config['eos_token_id'] = tokenizer.sep_token_id
    model_config['sep_token_id'] = tokenizer.sep_token_id
    model_config = RobertaConfig.from_dict(model_config)

    model = RobertaForMaskedLM(model_config)
    return tokenizer, model


def load_batches(
    filepaths: List[str],
    args: argparse.Namespace
) -> List[Batch]:
    texts = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8', newline='\n') as f:
            texts.extend([line.strip() for line in f])

    batcher = BatcherForLanguageModeling.from_args(args)
    return batcher(texts)


def _main(args: argparse.Namespace):
    assert torch.cuda.is_available(), 'Don\'t even try using CPU for this.'

    train_args = load_training_arguments(
        local_rank=args.local_rank,
        filepath=args.training_args,
    )

    tokenizer, model = load_tokenizer_and_model(
        tokenizer_path=args.tokenizer_path,
        model_config_path=args.model_config
    )

    eval_batches = None
    if args.eval_data_files is not None:
        eval_batches = load_batches(filepaths=args.eval_data_files, args=args)

    trainer = TrainerForMaskedLM(
        model=model,
        args=train_args,
        dataset_host=args.dataset_host,
        tokenizer=tokenizer,
        eval_batches=eval_batches,
    )
    trainer.train()


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description='Pre-training LM using roberta method.',
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_options(parser)
    _main(parser.parse_args(argv))


if __name__ == "__main__":
    main()
