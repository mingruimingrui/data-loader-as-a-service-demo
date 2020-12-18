"""Pre-train roberta model."""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Optional, Tuple

import requests
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.training_args import TrainingArguments
from transformers_pretraining.batching import Batch, BatcherForLanguageModeling


def add_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--model-config', required=True,
        help='Path to model config file.')
    parser.add_argument(
        '-t', '--tokenizer-path', required=True,
        help='Path to saved PretrainedTokenizer.')
    parser.add_argument(
        '--dataset-host', required=True,
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


def save_states(
    step_nb: int,
    train_args: TrainingArguments,
    model: RobertaForMaskedLM,
    optimizer: torch.optim.Optimizer
):
    os.makedirs(train_args.output_dir, exist_ok=True)
    prefix = os.path.join(train_args.output_dir, f'checkpoint.{step_nb}')

    model.save_pretrained(prefix + '.model')
    torch.save(optimizer, prefix + '.optimizer.pt')


def make_optimizer_and_lr_scheduler(
    model: torch.nn.Module,
    train_args: TrainingArguments,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_args.warmup_steps,
        num_training_steps=train_args.max_steps,
    )

    return optimizer, lr_scheduler


def query_batch(dataset_host: str) -> Batch:
    data = requests.get(dataset_host, stream=True).content
    return Batch.from_bytes(data)


def visualize_model_output(
    tokenizer: PreTrainedTokenizerFast,
    batch: Batch,
    model_output: MaskedLMOutput,
    n: int = 3,
    k: int = 5,
) -> str:
    """Visualize the inference output of a sample.

    Args:
        batch (Batch): The input batch.
        model_output (MaskedLMOutput): Model output.
        n (int, optional): Maximum number of samples visualized. Defaults to 3.
        k (int, optional): Maximum number of predictions per word visualized.
            Defaults to 5.

    Returns:
        str: A markdown formatted string.
    """
    n = min(len(batch.idxs), n)
    k = min(model_output.logits.shape[-1], k)

    logits = model_output.logits.detach().cpu()

    results = []

    for entry_id in range(n):
        orig_text = batch.texts[entry_id]
        input_ids = batch.input_ids[entry_id]
        probs = torch.softmax(logits[entry_id], -1)

        result = f'Sample entry #{entry_id + 1} : `{orig_text}`<br>\n'
        result += '| Token | Preds |\n'
        result += '| ----- | ----- |\n'

        for input_id, probs in zip(input_ids, probs):
            token = tokenizer._convert_id_to_token(input_id)
            topk_probs, topk_token_ids = torch.topk(probs, dim=-1)
            topk_tokens = tokenizer.convert_ids_to_tokens(topk_token_ids)
            topk_probs = [round(float(p), 2) for p in topk_probs]

            result += '| `{}` | `{}` |\n'.format(
                token,
                list(zip(topk_tokens, topk_probs))
            )

        results.append(result)

    return '\n\n'.join(results)


def main(args: argparse.Namespace):
    assert torch.cuda.is_available(), 'Don\'t even try using CPU for this.'

    # Load and initialite models and other training/logging related objects
    train_args = load_training_arguments(
        local_rank=args.local_rank,
        filepath=args.training_args,
    )

    is_distributed = torch.distributed.is_initialized()
    is_main_thread = not is_distributed or torch.distributed.get_rank() == 0

    if is_main_thread:
        os.makedirs(train_args.output_dir, exist_ok=True)
        writer = SummaryWriter(train_args.logging_dir)
        writer.add_text('Training Config', train_args.to_json_string())

    tokenizer, model = load_tokenizer_and_model(
        tokenizer_path=args.tokenizer_path,
        model_config_path=args.model_config
    )
    model = model.cuda()
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[train_args.local_rank],
            output_device=train_args.local_rank,
            broadcast_buffers=False
        )

    optimizer, lr_scheduler = make_optimizer_and_lr_scheduler(
        model=model, train_args=train_args)
    scaler = torch.cuda.amp.GradScaler(enabled=train_args.fp16)

    if is_main_thread:
        pbar = tqdm(total=train_args.gradient_accumulation_steps)

    # Define step
    def step():
        batch = query_batch(args.dataset_host)

        input_ids = torch.LongTensor(batch.input_ids).cuda()
        attention_mask = torch.LongTensor(batch.attention_mask).cuda()
        labels = torch.LongTensor(batch.labels).cuda()

        with torch.cuda.amp.autocast():
            model_output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        if torch.isnan(model_output.loss):
            return None

        scaler.scale(model_output.loss).backward()

        return batch, model_output

    def step_and_update_metrics(metrics: dict):
        if is_main_thread:
            pbar.update(1)

        try:
            sample = step()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                metrics['n_oom'] += 1
                return None
            raise e

        if sample is None:
            return None
        batch, model_output = sample

        with torch.no_grad():
            metrics['loss'] += model_output.loss.detach()
            metrics['n_batches'] += 1
            metrics['n_sents'] += len(batch.idxs)
            metrics['n_tokens'] += batch.attention_mask.sum()

        return batch, model_output

    for step_nb in range(1, train_args.max_steps + 1):
        # Initialize/reset logging variables
        is_final_step = step_nb == train_args.max_steps
        step_start_time = time.time()

        metrics = defaultdict(float)
        metrics['step_nb'] = step_nb
        metrics['n_oom'] = 0

        if is_main_thread:
            pbar.reset()
            pbar.set_description(f'Step: {step_nb}')

        # Pre gradient update
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Gradient accumulate
        last_valid_sample = None
        if is_distributed:
            with model.no_sync():
                for _ in range(train_args.gradient_accumulation_steps - 1):
                    sample = step_and_update_metrics(metrics)
                    if sample is not None:
                        last_valid_sample = sample
            sample = step_and_update_metrics(metrics)
            if sample is not None:
                last_valid_sample = sample
        else:
            for _ in range(train_args.gradient_accumulation_steps):
                sample = step_and_update_metrics(metrics)
                if sample is not None:
                    last_valid_sample = sample

        # Post gradient update
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # All the post gradient update stuff on main thread only
        if not is_main_thread:
            continue

        if metrics['n_batches'] == 0:
            raise RuntimeError('Did not manage to train a single batch')

        # Log metrics to tensorboard
        step_time_taken = time.time() - step_start_time
        metrics['loss'] = float(metrics['loss']) / metrics['n_batches']
        metrics['p_oom'] = float(metrics['n_oom']) / \
            train_args.gradient_accumulation_steps
        metrics['avg_token/sent'] = metrics['n_tokens'] / metrics['n_sents']
        metrics['batch/s'] = metrics['n_batches'] / step_time_taken
        metrics['token/s'] = metrics['n_tokens'] / step_time_taken
        metrics['sent/s'] = metrics['n_sents'] / step_time_taken
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step=step_nb)
        writer.flush()

        should_save = step_nb % train_args.save_steps == 0 or is_final_step
        if should_save:
            save_states(
                step_nb=step_nb,
                train_args=train_args,
                model=model.module if is_distributed else model,
                optimizer=optimizer,
            )

        should_log = step_nb % train_args.logging_steps == 0 or is_final_step
        if should_log and last_valid_sample is not None:
            batch, model_output = last_valid_sample
            result = visualize_model_output(
                tokenizer=tokenizer,
                batch=batch,
                model_output=model_output,
            )
            writer.add_text('training_sample', result, global_step=step_nb)
            writer.flush()

    if is_main_thread:
        pbar.close()
        writer.close()


if __name__ == "__main__":
    main(add_options(argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )).parse_args())
