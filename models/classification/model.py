# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import (
    absolute_import,
    division,
    print_function,
)

import argparse
import glob
import logging
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from features import convert_examples_to_features
import hypertune
from settings import TqdmLoggingHandler
# from knockknock import slack_sender

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def get_mismatched(labels, args, processor, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args.data_dir)
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    return wrong


def get_eval_report(labels, preds, args, processor):
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    acc = simple_accuracy(preds, labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return (
        {
            "mcc": mcc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "f1": f1,
            "acc": acc,
            "acc_and_f1": (acc + f1) / 2,
            "prec": prec,
            "recall": recall,
        },
        get_mismatched(labels, args, processor, preds),
    )


def compute_metrics(
    task_name, preds, labels, args, processor
):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds, args, processor)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizer,
    ),
    "xlnet": (
        XLNetConfig,
        XLNetForSequenceClassification,
        XLNetTokenizer,
    ),
    "xlm": (
        XLMConfig,
        XLMForSequenceClassification,
        XLMTokenizer,
    ),
    "roberta": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    ),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# @slack_sender(
#     webhook_url=os.environ["SLACK_WEBHOOK_URL"], channel=os.environ["SLACK_CHANNEL"]
# )
def train(args, train_dataset, model, processor, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = (
        args.per_gpu_train_batch_size * max(1, args.n_gpu)
    )
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (
                len(train_dataloader)
                // args.gradient_accumulation_steps
            )
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.per_gpu_train_batch_size,
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (
            torch.distributed.get_world_size()
            if args.local_rank != -1
            else 1
        ),
    )
    logger.info(
        "  Gradient Accumulation steps = %d",
        args.gradient_accumulation_steps,
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(
        args
    )  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2]
                    if args.model_type in ["bert", "xlnet"]
                    else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = (
                    loss.mean()
                )  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = (
                    loss / args.gradient_accumulation_steps
                )

            if args.fp16:
                with amp.scale_loss(
                    loss, optimizer
                ) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (
                (step + 1)
                % args.gradient_accumulation_steps
                == 0
                and not args.tpu
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer),
                        args.max_grad_norm,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps
                    == 0
                ):
                    logs = {}
                    # Log metrics
                    if (
                        args.local_rank == -1
                        and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(
                            args,
                            model,
                            processor,
                            tokenizer,
                        )
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir,
                        "checkpoint-{}".format(global_step),
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module
                        if hasattr(model, "module")
                        else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(
                        output_dir
                    )
                    torch.save(
                        args,
                        os.path.join(
                            output_dir, "training_args.bin"
                        ),
                    )
                    logger.info(
                        "Saving model checkpoint to %s",
                        output_dir,
                    )

            if args.tpu:
                args.xla_model.optimizer_step(
                    optimizer, barrier=True
                )
                model.zero_grad()
                global_step += 1

            if (
                args.max_steps > 0
                and global_step > args.max_steps
            ):
                epoch_iterator.close()
                break
        if (
            args.max_steps > 0
            and global_step > args.max_steps
        ):
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

# @slack_sender(
#     webhook_url=os.environ["SLACK_WEBHOOK_URL"], channel=os.environ["SLACK_CHANNEL"]
# )
def evaluate(args, model, processor, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        ("mnli", "mnli-mm")
        if args.task_name == "mnli"
        else (args.task_name,)
    )
    eval_outputs_dirs = (
        (args.output_dir, args.output_dir + "-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )
    hpt = hypertune.HyperTune()

    results = {}
    for eval_task, eval_output_dir in zip(
        eval_task_names, eval_outputs_dirs
    ):
        eval_dataset = load_and_cache_examples(
            args,
            processor,
            eval_task,
            tokenizer,
            evaluate=True,
        )

        if not os.path.exists(
            eval_output_dir
        ) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = (
            args.per_gpu_eval_batch_size
            * max(1, args.n_gpu)
        )
        # Note that DistributedSampler samples randomly
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
        )

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
            
        # Eval!
        logger.info(
            "***** Running evaluation {} *****".format(
                prefix
            )
        )
        logger.info(
            "  Num examples = %d", len(eval_dataset)
        )
        logger.info(
            "  Batch size = %d", args.eval_batch_size
        )
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(
            eval_dataloader, desc="Evaluating"
        ):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                        if args.model_type
                        in ["bert", "xlnet"]
                        else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = (
                    inputs["labels"].detach().cpu().numpy()
                )
            else:
                preds = np.append(
                    preds,
                    logits.detach().cpu().numpy(),
                    axis=0,
                )
                out_label_ids = np.append(
                    out_label_ids,
                    inputs["labels"].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        # TODO what to do with wrongs?
        result, wrong = compute_metrics(
            eval_task, preds, out_label_ids, args, processor
        )
        results.update(result)

        output_eval_file = os.path.join(
            eval_output_dir, prefix, "eval_results.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info(
                "***** Eval results {} *****".format(prefix)
            )
            for key in sorted(result.keys()):
                hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=key,
                    metric_value=result[key],
                )
                logger.info(
                    "  %s = %s", key, str(result[key])
                )
                writer.write(
                    "%s = %s\n" % (key, str(result[key]))
                )

    return results


def load_and_cache_examples(
    args, processor, task, tokenizer, evaluate=False
):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    output_mode = args.output_mode
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(
                filter(
                    None, args.model_name_or_path.split("/")
                )
            ).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if (
        os.path.exists(cached_features_file)
        and not args.overwrite_cache
    ):
        logger.info(
            "Loading features from cached file %s",
            cached_features_file,
        )
        features = torch.load(cached_features_file)
    else:
        logger.info(
            "Creating features from dataset file at %s",
            args.data_dir,
        )
        label_list = processor.get_labels()

        examples = (
            processor.get_dev_examples(args.data_dir)
            if evaluate
            else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples=examples,
            label_list=label_list,
            max_seq_length=args.max_seq_length,
            seq_func_params=args.seq_func_params,
            tokenizer=tokenizer,
            output_mode=output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
            seq_len_func=args.seq_len_func
        )
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file %s",
                cached_features_file,
            )
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long
    )
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features],
        dtype=torch.long,
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long,
    )
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long
        )
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float
        )

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels,
    )
    return dataset
