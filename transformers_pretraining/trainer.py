import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from transformers_pretraining.batching import Batch
from transformers_pretraining.cached_generator import CachedGenerator


def remove_file_or_dir(path: str):
    if not os.path.exists(path):
        return
    if os.path.isfile():
        os.remove(path)
    elif os.path.isdir():
        shutil.rmtree(path)


@dataclass
class StepMetrics:
    n_oom: int = 0
    n_nan_loss: int = 0
    total_loss: torch.FloatTensor = 0
    n_batches: int = 0
    n_sents: int = 0
    n_tokens: int = 0


class TrainerForMaskedLM:
    """A Masked LM trainer that uses a data-loading service.

    The usage pattern of this class would typically look like this.

    ```
    trainer = TrainerForMaskedLM(model, args, dataset_host)
    trainer.train()
    ```

    Random Notes:
    This class is really just used as a namespace.
    It does not have to be an object, in fact many would argue that it can
    be written as a function instead.

    Consider the usage example above.
    There really is nothing else to do with the `trainer` beyond calling the
    `train()` function.

    But god does it make it easy to write training code.
    This "namespace" essentially acts as a scope for storing objects for
    checkpointing, logging, evaluation which would otherwise be tedious to pass
    around.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        dataset_host: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        eval_batches: Optional[List[Batch]] = None,
        optimizers: Tuple[
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LambdaLR
        ] = (None, None)
    ):
        assert not isinstance(model, DistributedDataParallel), (
            'Trainer would handle wrapping of model in DDP for you. '
            'Don\'t pre-wrap so I won\'t have to write integration code.'
        )
        model = model.to(args.device)

        self.model = model
        self.args = args
        self.dataset_host = dataset_host
        self.tokenizer = tokenizer
        self.eval_batches = eval_batches
        self.optimizer, self.lr_scheduler = optimizers

        self.init_optimizer_and_lr_scheduler()
        self.batch_generator = CachedGenerator(self.make_batch_generator())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        self.is_distributed = torch.distributed.is_initialized()
        self.is_main_thread = (
            not self.is_distributed or
            torch.distributed.get_rank() == 0
        )

        self.ddp_model = None
        if self.is_distributed:
            self.ddp_model = DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False,
            )

        self.checkpoint_files = []
        self.pbar: Optional[tqdm] = None
        self.ministep_pbar: Optional[tqdm] = None
        self.writer: Optional[SummaryWriter] = None

    def init_optimizer_and_lr_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer is None:
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.max_steps,
            )

    def make_batch_generator(self):
        while True:
            data = requests.get(self.dataset_host, stream=True).content
            batch = Batch.from_bytes(data)
            yield batch

    def save_model(self, step_nb: int):
        os.makedirs(self.args.output_dir, exist_ok=True)
        prefix = os.path.join(self.args.output_dir, f'checkpoint.{step_nb}')
        model_save_path = f'{prefix}.model'
        optimizer_save_path = f'{prefix}.optimizer.pt'

        self.model.save_pretrained(model_save_path)
        torch.save(self.optimizer, optimizer_save_path)

        self.checkpoint_files.append((model_save_path, optimizer_save_path))
        save_total_limit = self.args.save_total_limit
        if (
            save_total_limit is not None and
            len(self.checkpoint_files) > save_total_limit
        ):
            files_to_delete = self.checkpoint_files[:-save_total_limit]
            files_to_keep = self.checkpoint_files[-self.args.save_total_limit:]

            for model_save_path, optimzier_save_path in files_to_delete:
                remove_file_or_dir(model_save_path)
                remove_file_or_dir(optimizer_save_path)

            self.checkpoint_files = files_to_keep

    def log_metrics(self, step_nb: int, step_metrics: StepMetrics):
        if self.writer is None:
            return

        self.writer.add_scalar(
            tag='loss',
            scalar_value=step_metrics.total_loss / step_metrics.n_batches,
            global_step=step_nb
        )
        for tag in [
            'n_oom', 'n_nan_loss',
            'n_batches', 'n_sents', 'n_tokens'
        ]:
            self.writer.add_scalar(
                tag, getattr(step_metrics, tag),
                global_step=step_nb
            )
        self.writer.flush()

    @torch.no_grad()
    def eval(self, step_nb: int):
        # Perhaps this should be named eval and visualize?

        if self.eval_batches is None or self.writer is None:
            return

        total_loss = 0
        for batch in self.eval_batches:
            model_inputs = batch.to_model_inputs(self.args.device)
            model_output = self.model.forward(**model_inputs, return_dict=True)
            total_loss += model_output.loss.detach()
        loss = total_loss / len(self.eval_batches)

        self.writer.add_scalar('eval_loss', loss, global_step=step_nb)
        self.writer.flush()

        self.visualize(step_nb, batch, model_output)

    @torch.no_grad()
    def visualize(
        self,
        step_nb: int,
        batch: Batch,
        model_output: MaskedLMOutput,
        n: int = 3,
        k: int = 5,
    ):
        if self.writer is None or self.tokenizer is None:
            return

        n = min(len(batch.idxs), n)
        k = min(model_output.logits.shape[-1], k)
        logits = model_output.logits.detach().cpu()

        msgs = []
        for i in range(n):
            orig_text = batch.texts[i]
            input_ids = batch.input_ids[i]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            probs = torch.softmax(logits[i], -1)

            msg = f'## Sample entry #{i + 1}: `{orig_text}`\n\n'
            msg += '| Token | Preds |\n'
            msg += '| ----- | ----- |\n'

            for input_token, probs in zip(input_tokens, probs):
                topk_probs, topk_token_ids = torch.topk(probs, k, dim=-1)
                topk_tokens = \
                    self.tokenizer.convert_ids_to_tokens(topk_token_ids)
                topk_probs = [round(float(p), 2) for p in topk_probs]

                msg += '| `{}` | `{}` |\n'.format(
                    input_token,
                    list(zip(topk_tokens, topk_probs))
                )
            msgs.append(msg)

        self.writer.add_text('output', '\n'.join(msgs), global_step=step_nb)
        self.writer.flush()

    def ministep(self) -> Tuple[Batch, Optional[MaskedLMOutput], bool]:
        """Ministep is for computing and accumulating the gradients."""
        batch = next(self.batch_generator)
        model_inputs = batch.to_model_inputs(self.args.device)
        model = self.model if self.ddp_model is None else self.ddp_model

        with torch.cuda.amp.autocast():
            try:
                model_output = model.forward(
                    **model_inputs,
                    return_dict=True
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    return batch, None, True
                raise e

        if torch.isnan(model_output.loss):
            return batch, None, False

        self.scaler.scale(model_output.loss).backward()
        return batch, model_output, False

    def ministep_callback(
        self,
        step_metrics: StepMetrics,
        batch: Batch,
        model_output: Optional[MaskedLMOutput],
        oom: bool,
    ):
        """Ministep callback accumulates metrics."""
        if self.ministep_pbar is not None:
            self.ministep_pbar.update(1)

        if oom:
            step_metrics.n_oom += 1

        elif model_output is None:
            step_metrics.n_nan_loss += 1

        else:
            step_metrics.total_loss += model_output.loss.detach()
            step_metrics.n_batches += 1
            step_metrics.n_sents += len(batch.idxs)
            step_metrics.n_tokens += batch.attention_mask.sum()

    def step(self, step_nb: int) -> StepMetrics:
        """Each step consists of multiple ministeps + gradient update."""
        # Initialize progress and metrics
        step_metrics = StepMetrics()
        if self.ministep_pbar is not None:
            self.ministep_pbar.reset()
            self.ministep_pbar.set_description(f'Step {step_nb}')

        # Pre-gradient accumulation setting up
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Accumulate gradients
        if self.ddp_model is not None:
            with self.ddp_model.no_sync():
                for _ in range(self.args.gradient_accumulation_steps - 1):
                    self.ministep_callback(step_metrics, *self.ministep())
            self.ministep_callback(step_metrics, *self.ministep())
        else:
            for _ in range(self.args.gradient_accumulation_steps):
                self.ministep_callback(step_metrics, *self.ministep())

        # Update gradients, optimizer, scaler, lr scheduler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

        return step_metrics

    def callback(self, step_nb: int, step_metrics: StepMetrics):
        """Handling of logging, model saving, eval and visualization."""
        if self.pbar is not None:
            self.pbar.update(1)

        is_first_step = step_nb == 1
        is_final_step = step_nb == self.args.max_steps
        should_save_step = step_nb % self.args.save_steps == 0 or is_final_step
        should_log_step = (
            step_nb % self.args.logging_steps == 0 or
            (self.args.logging_first_step and is_first_step)
        )
        should_eval_step = step_nb % self.args.eval_steps == 0 or is_final_step

        if should_save_step and self.is_main_thread:
            self.save_model(step_nb)

        if should_log_step and self.writer is not None:
            self.log_metrics(step_nb, step_metrics)

        if (
            should_eval_step and
            self.eval_batches is not None and
            self.writer is not None
        ):
            self.eval(step_nb)

    def train(self):
        if self.is_main_thread:
            self.pbar = tqdm(desc='Training', total=self.args.max_steps)
            self.ministep_pbar = \
                tqdm(total=self.args.gradient_accumulation_steps)
            self.writer = SummaryWriter(self.args.logging_dir)

        for step_nb in range(1, self.args.max_steps + 1):
            step_metrics = self.step(step_nb)
            self.callback(step_nb, step_metrics)

        if self.is_main_thread:
            self.pbar.close()
            self.ministep_pbar.close()
            self.writer.close()
